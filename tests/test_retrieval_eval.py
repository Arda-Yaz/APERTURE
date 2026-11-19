import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import isolation_support
from isolation_support import TestTemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import retrieval_eval


class RetrievalEvaluationTests(unittest.TestCase):
    def test_offline_report_measures_each_production_path(self):
        retrieval_eval.SCRATCH.mkdir(exist_ok=True)
        with TestTemporaryDirectory(dir=retrieval_eval.SCRATCH) as directory:
            output = Path(directory) / 'report.json'
            report = retrieval_eval.evaluate(output, group='fresh')
            self.assertEqual(report, json.loads(output.read_text(encoding='utf-8')))
            with self.assertRaises(FileExistsError):
                retrieval_eval.evaluate(output, group='historical')
            self.assertEqual(report, json.loads(output.read_text(encoding='utf-8')))

        self.assertEqual(report['model_calls'], 0)
        self.assertIsNone(report['model'])
        self.assertEqual(report['actor_selector_mode'], 'gold_id_mock')
        self.assertEqual(report['guard_violations'], [])
        self.assertEqual(report['source_sha256']['app/memory.py'],
                         hashlib.sha256((ROOT / 'app/memory.py').read_bytes()).hexdigest())
        self.assertEqual(report['fixture_sha256'], {
            'evaluations/retrieval/baseline.json':
                '904fba12cece60c9a3c1b42b77e68d66ce8a66d33ed551a576bd34011fab91c3',
            'evaluations/retrieval/fresh.json':
                '1621da914d23ca99688a7df2d5c28a01d43f8a6c58d1302f4840e67489586786',
        })
        paths = report['groups']['fresh']
        self.assertEqual(set(paths), {'lexical_tool', 'actor_fallback', 'actor_selector'})
        for item in paths.values():
            metrics = item['metrics']
            self.assertEqual(metrics['completed_measurements'], 10)
            self.assertEqual(metrics['recall_at_5']['denominator'], 6)
            self.assertEqual(metrics['no_answer_false_positive_rate']['denominator'], 4)
            self.assertEqual(metrics['inactive_returns'], 0)
            self.assertEqual(metrics['ownership_tag_violations'], 0)
        self.assertIsNone(paths['lexical_tool']['results'][0]['candidate_ids'])
        self.assertTrue(paths['actor_fallback']['results'][0]['candidate_ids'])
        for row in paths['actor_selector']['results']:
            self.assertEqual(row['returned_ids'],
                             [key for key in row['gold_ids'] if key in row['candidate_ids']])

    def test_local_digest_requires_the_requested_model_tag(self):
        models = [SimpleNamespace(model='fixture:latest', digest='one'),
                  SimpleNamespace(model='fixture:tested', digest='two')]
        client = SimpleNamespace(list=lambda: SimpleNamespace(models=models))
        self.assertEqual(retrieval_eval.installed_digest(client, 'fixture'), 'one')
        self.assertEqual(retrieval_eval.installed_digest(client, 'fixture:tested'), 'two')
        with self.assertRaisesRegex(ValueError, 'no installed digest'):
            retrieval_eval.installed_digest(client, 'fixture:untested')

    def test_live_route_records_digest_and_uses_only_bounded_retrieval_calls(self):
        isolation_support.install_guards()
        import ollama

        clients = []

        class LocalClientDouble:
            def __init__(self, **options):
                clients.append(self)
                self.options = options
                self.inherited_key = os.environ.get('OLLAMA_API_KEY')
                self.requests = []
                self.closed = False

            def list(self):
                return SimpleNamespace(models=[SimpleNamespace(model='fixture:local', digest='digest')])

            def chat(self, **request):
                self.requests.append(request)
                return SimpleNamespace(message=SimpleNamespace(content='{"memory_ids": []}'))

            def close(self):
                self.closed = True

        retrieval_eval.SCRATCH.mkdir(exist_ok=True)
        with TestTemporaryDirectory(dir=retrieval_eval.SCRATCH) as directory, \
                patch.object(ollama, 'Client', LocalClientDouble), \
                patch.dict(os.environ, OLLAMA_API_KEY='fixture-key'), \
                patch.object(isolation_support, '_INSTALLED', False), \
                patch.object(isolation_support, 'install_guards'):
            # Stand in for the standalone CLI's initialization. The storage
            # guards stay installed and the client above cannot use a network.
            report = retrieval_eval.evaluate(Path(directory) / 'report.json',
                                             live=True, model='fixture:local', group='fresh')
            self.assertEqual(os.environ['OLLAMA_API_KEY'], 'fixture-key')

        client, = clients
        self.assertTrue(client.closed)
        self.assertIsNone(client.inherited_key)
        self.assertEqual(client.options['host'], 'http://127.0.0.1:11434')
        self.assertFalse(client.options['trust_env'])
        self.assertFalse(client.options['follow_redirects'])
        self.assertEqual(report['model']['digest'], 'digest')
        self.assertEqual(report['model_calls'], len(client.requests))
        self.assertEqual(len(client.requests), report['model']['max_calls'])
        self.assertEqual(len(client.requests), 10)
        self.assertTrue(all(request['model'] == 'fixture:local' and request['format'] == 'json'
                            for request in client.requests))
        self.assertEqual(report['groups']['fresh']['actor_selector']['metrics']['selector_errors'], 0)
        self.assertEqual(report['guard_violations'], [])


if __name__ == '__main__':
    unittest.main()
