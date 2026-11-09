import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mixed_pilot_mtrouter.audit import audit_directories
from mixed_pilot_mtrouter.splitting import split_indices
from mixed_pilot_mtrouter.service import score_request


class CoreTests(unittest.TestCase):
    def test_group_split_keeps_models_and_runs_together(self):
        samples = [{'benchmark':'scienceworld', 'task_id':str(task), 'model_idx':m}
                   for task in range(5) for m in range(3) for step in range(3)]
        a,b = split_indices(samples,0.2)
        self.assertTrue(a and b)
        self.assertFalse({samples[i]['task_id'] for i in a} & {samples[i]['task_id'] for i in b})
        self.assertEqual(sorted(a+b),list(range(len(samples))))
        self.assertEqual((a,b),split_indices(samples,0.2))

    def test_service_rejects_order_and_nonfinite_values(self):
        class Fake:
            model_names=['a','b']
            def predict(self,*args):return [1,float('nan')]
        with self.assertRaisesRegex(ValueError,'order'):
            score_request(Fake(),{'models':['b','a']})
        with self.assertRaisesRegex(ValueError,'invalid Q'):
            score_request(Fake(),{'models':['a','b'],'messages':[{'role':'user','content':'x'}]})

    def test_offline_precompute_train_reload(self):
        import torch
        import yaml
        from transformers import BertConfig, BertModel, BertTokenizerFast
        from mixed_pilot_mtrouter.research.training.q_network import NeuralQFunction
        from mixed_pilot_mtrouter.cli.train_q_function import load_training_config
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); model=root/'encoder';model.mkdir();data=root/'data';data.mkdir()
            vocab=model/'vocab.txt';vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nwater\ntask\nlook\n')
            BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(model)
            # Production embedding weights may be stored as BF16. Online
            # inference must still use the same FP32 encoder as precomputation.
            BertModel(BertConfig(vocab_size=8,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32,max_position_embeddings=128)).to(dtype=torch.bfloat16).save_pretrained(model)
            names=['qwen-flash-2025-07-28','deepseek-flash']
            for task in range(4):
                for idx,name in enumerate(names):
                    messages=[{'role':'user','content':f'<task_description>water task {task}</task_description>'}]
                    for step in range(2):
                        messages.extend([{'role':'assistant','content':'look','model_name':name,'propensity':1.0,'extra':{'response':{'usage':{'cost':0.01}}}}, {'role':'user','content':'water'}])
                    (data/f'{task}-{idx}.traj.json').write_text(json.dumps({'info':{'benchmark':'scienceworld','task_id':f'task{task}','final_score':100 if idx==0 else -100,'cost_currency':'CNY'},'messages':messages}))
            audit=audit_directories([data],names);self.assertEqual(audit['steps'],16)
            with self.assertRaisesRegex(ValueError,'unknown model'): audit_directories([data],['wrong'])
            cfg=load_training_config();cfg['history_encoder'].update(model_name=str(model),max_tokens=64,backend='hf',pooling_mode='cls')
            cfg['q_network']={'hidden_dims':[16],'dropout':0}
            path=root/'config.yaml';path.write_text(yaml.safe_dump(cfg));emb=root/'embedding.pt';out=root/'output'
            env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1'}
            for args in [
                ['mixed_pilot_mtrouter.cli.precompute_embeddings','--config',str(path),'--data-dir',str(data),'--output',str(emb),'--backend','hf'],
                ['mixed_pilot_mtrouter.cli.train_q_function','--config',str(path),'--data-dir',str(data),'--output-dir',str(out),'--precomputed-path',str(emb),'--epochs','1','--batch-size','4','--val-ratio','0.25'],
            ]:
                result=subprocess.run([sys.executable,'-m',*args,'--device','cpu'],capture_output=True,text=True,env=env,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout[-6000:]+result.stderr[-6000:])
            checkpoint=out/'q_function_best.pt'; self.assertTrue(checkpoint.exists())
            loaded_names, config, tradeoff=NeuralQFunction.load_config_from_checkpoint(checkpoint)
            self.assertEqual(loaded_names,names)
            q=NeuralQFunction(names,config=config,lambda_=tradeoff);q.load(checkpoint)
            self.assertEqual(next(q.history_encoder.encoder.parameters()).dtype,torch.float32)
            from mixed_pilot_mtrouter.service import load_scorer
            local=load_scorer(checkpoint, expected_models=names)
            with self.assertRaisesRegex(ValueError,'Model names mismatch'):
                load_scorer(checkpoint, expected_models=list(reversed(names)))
            messages=[{'role':'user','content':'water task'}]
            response=score_request(q,{'models':names,'messages':messages})
            self.assertEqual(score_request(local,{'models':names,'messages':messages}), response)
            self.assertIn(response['model'],names)
            # The frozen online encoder and precomputed features must agree.
            cache=torch.load(emb,weights_only=True)
            raw=json.loads((data/'0-0.traj.json').read_text())
            online=q.history_encoder(messages=raw['messages'][:1]).detach()
            self.assertTrue(torch.allclose(online,cache['embeddings'][0],atol=1e-5))
            q.train();self.assertFalse(q.history_encoder.encoder.training)

            # Exercise the real HTTP endpoint with the trained scorer.
            from mixed_pilot_mtrouter.service import make_server
            from urllib.request import Request, build_opener, ProxyHandler
            import threading
            server=make_server(q,port=0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                req=Request(f'http://127.0.0.1:{server.server_port}/route',
                            data=json.dumps({'models':names,'messages':messages}).encode(),
                            headers={'Content-Type':'application/json'})
                with build_opener(ProxyHandler({})).open(req,timeout=10) as http:
                    self.assertEqual(json.load(http)['model'],response['model'])
            finally:
                server.shutdown();server.server_close();thread.join(timeout=5)
            saved=root/'roundtrip.pt';q.save(saved)
            q2=NeuralQFunction(names,config=config);q2.load(saved)
            self.assertTrue(torch.allclose(torch.tensor(q.predict(messages,names)),torch.tensor(q2.predict(messages,names))))
            split=json.loads((out/'split.json').read_text())
            self.assertFalse(set(split['train_tasks']) & set(split['validation_tasks']))
            # Currency mix must fail before doing expensive work.
            f=next(data.glob('*.json'));d=json.loads(f.read_text());d['info']['cost_currency']='USD';f.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'Mixed currencies'):audit_directories([data],names)

if __name__=='__main__':unittest.main()
