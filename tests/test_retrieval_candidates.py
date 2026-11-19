import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from isolation_support import TestTemporaryDirectory, install_guards

install_guards()
import llm
from memory import MemoryStore


class RetrievalCandidateTests(unittest.TestCase):
    def test_recent_world_records_do_not_crowd_out_an_older_user_record(self):
        with TestTemporaryDirectory() as directory:
            store = MemoryStore(Path(directory) / 'memory.db')
            old_user_record = store.remember(
                content='Mira is most comfortable in quiet rooms.',
                category='fact', importance=3, subject='user', source='fixture',
            )
            for index in range(100):
                store.remember(
                    content=f'Inventory shelf {index} has a ceramic sample.',
                    category='fact', importance=3, subject='world', source='fixture',
                )

            supplied = []

            def select(**request):
                supplied.extend(request['candidates'])
                return [old_user_record['id']] if any(
                    record['id'] == old_user_record['id'] for record in supplied
                ) else []

            matches = store.search('Which acoustic setting supports focus?', selector=select)

        self.assertEqual([record['id'] for record in matches], [old_user_record['id']])
        self.assertEqual(matches[0]['subject'], 'user')
        self.assertLessEqual(len(supplied), 96)

    def test_retrieval_asks_ollama_to_generate_json(self):
        response = SimpleNamespace(message=SimpleNamespace(content='{"memory_ids": []}'))
        with patch.object(llm, 'ollama_chat', return_value=response) as model:
            result = llm._select_memory_ids(
                query='Which acoustic setting supports focus?',
                recent_dialogue='', candidates=[], limit=5,
            )
        self.assertEqual(result, [])
        self.assertEqual(model.call_args.kwargs['format'], 'json')
        self.assertEqual(model.call_args.kwargs['role'], 'retrieval')
