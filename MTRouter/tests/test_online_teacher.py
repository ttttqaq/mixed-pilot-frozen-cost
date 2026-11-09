"""Fixed Teacher labels, private references and Responses protocol; no network."""
import asyncio
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from mixed_pilot_harness.config import ProviderProfile
from mixed_pilot_harness.types import ProviderResponse, Usage, ProviderError
from mixed_pilot_mtrouter.online.config import DEFAULTS
from mixed_pilot_mtrouter.online.teacher import FixedReferenceTeacher, parse_estimate
from mixed_pilot_mtrouter.online.responses import ResponsesTeacherProvider, read_response
from mixed_pilot_mtrouter.online.runtime import InvalidTeacherEvidenceError
from mixed_pilot_mtrouter.online.references import load_references
from test_online_learning import state


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = deepcopy(DEFAULTS)
        self.reply = {"expected_reward": .7, "remaining_agent_cost": .02, "assessment_status": "ok"}
        self.runner = SimpleNamespace(ledger=SimpleNamespace(currency="CNY",records={}), pricing={"profiles": {n:{} for n in self.config["model_pool"]}},
            complete=AsyncMock(side_effect=lambda *a,**k: {"response":ProviderResponse(text=json.dumps(self.reply),stop_reason="end_turn", usage=Usage(10,10)),"call_id":"c1","cost":.01}))
        self.references = {"train--task":{"actions":["PRIVATE_GOLD_ACTION"],"sha256":"hash1"}}
        self.teacher = FixedReferenceTeacher(self.config,self.runner,self.root,self.references)
        self.context = {"version":"policy-0","epsilon":.1,"models":self.config["model_pool"]}

    def test_direct_targets_available_without_training_and_reference_is_in_teacher_request(self):
        value = asyncio.run(self.teacher.get(state(),self.context))
        self.assertEqual((value["reward"],value["cost"]),(.7,.02))
        self.assertTrue(value["ready"])
        request = json.loads(self.runner.complete.call_args.args[1][0].content)
        self.assertEqual(request["reference"]["actions"],["PRIVATE_GOLD_ACTION"])
        self.assertEqual(request["continuation_policy"]["epsilon"],.1)
        self.assertEqual(self.runner.complete.call_args.kwargs["tools"],[])
        self.assertEqual(self.runner.complete.call_args.kwargs["profile_overrides"]["extra_body"]["reasoning_effort"],"low")
        self.assertFalse(hasattr(self.teacher,"train"))

    def test_cached_value_persists_but_policy_reference_and_generation_change_invalidate_it(self):
        asyncio.run(self.teacher.get(state(),self.context))
        again = FixedReferenceTeacher(self.config,self.runner,self.root,self.references)
        asyncio.run(again.get(state(),self.context)); self.assertEqual(self.runner.complete.await_count,1)
        original = self.teacher.key(state(),self.context)
        self.references["train--task"]["sha256"]="hash2"
        self.assertNotEqual(original,self.teacher.key(state(),self.context))
        self.references["train--task"]["sha256"]="hash1"
        self.context["epsilon"]=1.
        self.assertNotEqual(original,self.teacher.key(state(),self.context))

    def test_quota_and_insufficient_information_do_not_create_labels(self):
        self.config["teacher"]["llm"]["max_calls_per_iteration"]=0
        self.assertIsNone(asyncio.run(self.teacher.get(state(),self.context)))
        self.runner.complete.assert_not_awaited()
        self.assertFalse(parse_estimate(json.dumps({**self.reply,"assessment_status":"insufficient_information"}))["ready"])

    def test_test_and_valid_states_cannot_reach_teacher(self):
        for split in ["test","valid"]:
            s=state();s["split"]=split
            with self.assertRaises(ValueError): asyncio.run(self.teacher.get(s,self.context))
        self.runner.complete.assert_not_awaited()

    def test_invalid_reply_is_recorded_and_resume_does_not_buy_it_again(self):
        self.reply["expected_reward"]=2
        with self.assertRaises(InvalidTeacherEvidenceError): asyncio.run(self.teacher.get(state(),self.context))
        self.assertTrue((self.root/'teacher_predictions.jsonl').exists())
        with self.assertRaises(InvalidTeacherEvidenceError): asyncio.run(self.teacher.get(state(),self.context))
        self.assertEqual(self.runner.complete.await_count,1)

    def test_invalid_target_fields_are_rejected(self):
        for key,value in [("expected_reward",True),("expected_reward",float('nan')),("remaining_agent_cost",-1)]:
            with self.assertRaises(ValueError): parse_estimate(json.dumps({**self.reply,key:value}))

    def test_evaluation_usage_never_enters_teacher_context(self):
        actor=self.config["model_pool"][0]
        self.runner.ledger.records={"test-call":{"profile":actor,"scope":"test","response":{"usage":{"input_tokens":999999,"output_tokens":99999}}},
            "train-call":{"profile":actor,"scope":"collection","response":{"usage":{"input_tokens":100,"output_tokens":10}}}}
        asyncio.run(self.teacher.get(state(),self.context))
        request=json.loads(self.runner.complete.call_args.args[1][0].content)
        self.assertEqual(request["observed_actor_usage"][actor]["mean_input_tokens"],100)

    def test_reference_loader_reads_train_only_and_detects_changed_content(self):
        d=self.root/'data';(d/'tasks/a/reference').mkdir(parents=True)
        (d/'train.json').write_text(json.dumps({"tasks":[{"id":"a","path":"tasks/a"}]}))
        (d/'test.json').write_text('MUST NOT READ')
        gold=d/'tasks/a/reference/gold_path.json';gold.write_text(json.dumps({"actions":["secret"],"completed":True,"final_score":100}))
        cfg={"dataset":{"root":str(d)}};task={"source_task_id":"a","task_id":"train--a","split":"train"};m={}
        first=load_references(cfg,[task],m);self.assertIn("train--a",first)
        self.assertNotIn('reference',task);digest=m["reference_fingerprint"]
        gold.write_text(json.dumps({"actions":["changed"],"completed":True,"final_score":100}))
        load_references(cfg,[task],m);self.assertNotEqual(digest,m["reference_fingerprint"])
        gold.unlink();gold.symlink_to(d/'test.json')
        with self.assertRaises(ValueError): load_references(cfg,[task],m)


class Reply(io.BytesIO):
    def __init__(self, result, stream=False):
        body=("data: "+json.dumps({"type":"response."+result["status"],"response":result})+"\r\n\r\n").encode() if stream else json.dumps(result).encode()
        super().__init__(body);self.headers={"Content-Type":"text/event-stream" if stream else "application/json"}


class ResponsesTests(unittest.TestCase):
    def result(self):
        return {"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"{}"}]}],
                "usage":{"input_tokens":100,"output_tokens":20,"input_tokens_details":{"cached_tokens":30},"output_tokens_details":{"reasoning_tokens":15}}}

    def test_json_and_sse_and_incomplete_terminal_are_preserved_for_billing(self):
        for stream in [True,False]: self.assertEqual(read_response(Reply(self.result(),stream))["usage"]["output_tokens"],20)
        value=self.result();value["status"]="incomplete"
        self.assertEqual(read_response(Reply(value,True))["status"],"incomplete")
        reply=io.BytesIO(b'data: {"type":"response.output_text.delta","delta":"partial"}\n\ndata: [DONE]\n\n');reply.headers={"Content-Type":"text/event-stream"}
        with self.assertRaises(ProviderError):read_response(reply)

    def test_sse_terminal_without_final_blank_and_delta_fallback(self):
        event={"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":1,"output_tokens":2}}}
        reply=io.BytesIO(('data: {"type":"response.output_text.delta","delta":"{}"}\n\n'+'data: '+json.dumps(event)).encode())
        reply.headers={"Content-Type":"text/event-stream"}
        value=read_response(reply)
        self.assertEqual(value["output"][0]["content"][0]["text"],"{}")

    def test_failed_request_cooldown_is_observed_before_retry(self):
        from urllib.error import HTTPError
        from mixed_pilot_harness.types import Message
        p=ProviderProfile(kind="online-responses-teacher",model="gpt-6.1-terra",base_url="https://code1.newcli.com/v1",api_key_env="NEWCLI_API_KEY")
        provider=ResponsesTeacherProvider(p,15.)
        async def run():
            with self.assertRaises(ProviderError): await provider.complete([Message("user","x")],[],"sys")
            return await provider.complete([Message("user","x")],[],"sys")
        with patch.dict('os.environ',NEWCLI_API_KEY='test-only'), patch('urllib.request.urlopen',side_effect=[HTTPError('https://invalid',429,'rate limit',{},io.BytesIO(b'wait')),Reply(self.result())]), patch('asyncio.sleep',new_callable=AsyncMock) as sleep:
            asyncio.run(run())
            self.assertEqual(sleep.await_count,2)
            self.assertGreater(sleep.call_args.args[0],14)

    def test_actual_payload_and_usage_follow_desktop_script_and_requests_are_spaced(self):
        from mixed_pilot_harness.types import Message
        profile=ProviderProfile(kind="online-responses-teacher",model="gpt-6.1-terra",base_url="https://code1.newcli.com/v1",api_key_env="NEWCLI_API_KEY",extra_body={"reasoning_effort":"low"})
        provider=ResponsesTeacherProvider(profile,min_interval=15.)
        with patch.dict('os.environ',NEWCLI_API_KEY='test-only'), patch('urllib.request.urlopen',side_effect=[Reply(self.result(),True),Reply(self.result())]) as send, patch('asyncio.sleep',new_callable=AsyncMock) as sleep:
            for _ in range(2): result=asyncio.run(provider.complete([Message("user","question")],[],"instructions"))
        payload=json.loads(send.call_args.args[0].data)
        self.assertEqual(send.call_args.args[0].full_url,'https://code1.newcli.com/v1/responses')
        self.assertEqual(payload["model"],"gpt-6.1-terra");self.assertEqual(payload["reasoning"],{"effort":"low"})
        self.assertEqual(payload["input"][0]["content"][0]["type"],"input_text")
        self.assertFalse(payload["store"]);self.assertTrue(payload["stream"])
        self.assertEqual(result.usage.output_tokens,20)
        self.assertEqual(result.usage.raw["prompt_tokens_details"]["cached_tokens"],30)
        self.assertEqual(sleep.call_count,2);self.assertGreater(sleep.call_args.args[0],14)
