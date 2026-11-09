from __future__ import annotations
import asyncio
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mixed_pilot_harness.config import ProviderProfile
from mixed_pilot_harness.router.config import load_router_config, RouterConfigError
from mixed_pilot_harness.router.provider import RouterProvider
from mixed_pilot_harness.router.opensquilla_v4 import RouterRuntimeError
from mixed_pilot_harness.types import Message, ProviderResponse, ToolDefinition


class BridgeTests(unittest.TestCase):
    def config(self, directory):
        path=Path(directory)/'router.toml'
        path.write_text('[router]\nengine="mtrouter"\nservice_url="http://127.0.0.1:8765"\n[tiers.small]\nprofile="a"\n[tiers.large]\nprofile="b"\n')
        return load_router_config(path,{'a':ProviderProfile(kind='scripted',model='model-a'),'b':ProviderProfile(kind='scripted',model='model-b')})

    def test_selected_provider_gets_original_request_and_decision_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=self.config(tmp);calls=[];requests=[]
            class Target:
                provider_name='test'
                def __init__(self,p):self.p=p
                async def complete(self,messages,tools,system):
                    calls.append((self.p.model,messages,tools,system));return ProviderResponse(text='ok')
            result={'model':'model-b','model_names':['model-a','model-b'],'q_values':{'model-a':0.1,'model-b':0.8}}
            def http(req,timeout):
                requests.append(json.loads(req.data));return io.BytesIO(json.dumps(result).encode())
            messages=[Message('user','task'),Message('assistant','look'),Message('user','feedback')]
            tools=[ToolDefinition('test','test',{})]
            with patch('mixed_pilot_harness.router.mtrouter.urlopen',side_effect=http):
                provider=RouterProvider(config,Target)
                response=asyncio.run(provider.complete(messages,tools,'system'))
            self.assertEqual(response.text,'ok');self.assertEqual(calls,[('model-b',messages,tools,'system')])
            self.assertEqual(requests[0]['messages'][0],{'role':'system','content':'system'})
            self.assertEqual(len(requests[0]['messages']),4)
            event=provider.drain_events()[0]
            self.assertEqual(event.data['decision_metadata']['q_values'],result['q_values'])
            self.assertFalse(event.data['decision_metadata']['confidence_available'])

    def test_invalid_response_does_not_call_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=self.config(tmp)
            for result in [
                {'model_names':['model-b','model-a']},
                {'model_names':['model-a','model-b'],'q_values':{'model-a':float('nan'),'model-b':0.8}},
                {'model_names':['model-a','model-b'],'model':'model-a','q_values':{'model-a':0.1,'model-b':0.8}},
            ]:
                class Target:
                    async def complete(self,*args):raise AssertionError('provider must not be called')
                provider=RouterProvider(config,lambda p:Target())
                with patch('mixed_pilot_harness.router.mtrouter.urlopen',return_value=io.BytesIO(json.dumps(result).encode())):
                    with self.assertRaises(RouterRuntimeError):asyncio.run(provider.complete([Message('user','x')],[],''))

if __name__=='__main__':unittest.main()
