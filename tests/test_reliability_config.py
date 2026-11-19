"""Day 1 configuration/client contracts; all model calls are mocked."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from isolation_support import install_guards
install_guards()

import config
import inference
import llm
import reflection
import relationship
import self_state
import task_controller


def response(content='answer', **message_fields):
    return SimpleNamespace(message=SimpleNamespace(content=content, **message_fields))


class ReliabilityConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.original_environment = dict(os.environ)
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_every_role_inherits_documented_default_and_experiments_stay_off(self):
        settings = config.resolved_inference_settings()
        self.assertEqual(set(settings), set(config.INFERENCE_ROLES))
        self.assertEqual({item['model'] for item in settings.values()}, {'qwen3.5:4b'})
        self.assertTrue(all(item['retries'] == 0 for item in settings.values()))
        self.assertFalse(config._env_flag('APERTURE_DYNAMIC_SELF'))
        self.assertFalse(config._env_flag('APERTURE_RELATIONSHIP'))
        self.assertFalse(config._env_flag('APERTURE_SELF_CONSOLIDATION'))

    def test_automatic_user_memory_requires_approval_unless_explicitly_disabled(self):
        name = 'APERTURE_REQUIRE_REFLECTION_USER_MEMORY_APPROVAL'
        startup_environment = {key: value for key, value in self.original_environment.items()
                               if key != name}
        for value, expected in ((None, 'True'), ('typo', 'True'), ('0', 'False')):
            with self.subTest(value=value):
                env = {} if value is None else {name: value}
                completed = subprocess.run(
                    [sys.executable, '-B', '-c',
                     'from config import REQUIRE_REFLECTION_USER_MEMORY_APPROVAL as policy; print(policy)'],
                    cwd=Path(config.__file__).resolve().parent,
                    env={**startup_environment, **env}, capture_output=True, text=True, check=True)
                self.assertEqual(completed.stdout.strip(), expected)

    def test_global_and_role_overrides_follow_precedence_without_mutating_environment(self):
        os.environ.update(APERTURE_MODEL='shared:custom', APERTURE_MODEL_REFLECTION='reflect:custom',
                          APERTURE_INFERENCE_TIMEOUT_SECONDS='12', APERTURE_INFERENCE_TIMEOUT_SECONDS_REFLECTION='3',
                          APERTURE_INFERENCE_RETRIES='1', APERTURE_INFERENCE_RETRIES_REFLECTION='2',
                          APERTURE_INFERENCE_NUM_PREDICT_REFLECTION='128')
        before = dict(os.environ)
        actor = config.resolve_inference_settings('actor')
        reflected = config.resolve_inference_settings('reflection')
        self.assertEqual((actor.model, actor.timeout_seconds, actor.retries), ('shared:custom', 12, 0))
        self.assertEqual((reflected.model, reflected.timeout_seconds, reflected.retries, reflected.num_predict),
                         ('reflect:custom', 3, 2, 128))
        self.assertEqual(config.resolve_inference_settings('relationship').model, 'shared:custom')
        self.assertEqual(dict(os.environ), before)

    def test_endpoint_compatibility_precedence_and_secret_free_diagnostics(self):
        os.environ['OLLAMA_HOST'] = 'localhost:12345'
        self.assertEqual(config.resolve_inference_settings().host, 'http://localhost:12345')
        os.environ['APERTURE_OLLAMA_HOST'] = 'https://user:password@localhost:2345/private-token?key=secret#fragment'
        resolved = config.resolve_inference_settings()
        self.assertIn('password', resolved.host)  # Use the intentional override intact.
        public = json.dumps(config.resolved_inference_settings())
        self.assertIn('https://localhost:2345', public)
        for secret in ('password', 'private-token', 'secret', 'fragment', 'user:'):
            self.assertNotIn(secret, public)

    def test_invalid_settings_rejected_without_echoing_values(self):
        cases = [('APERTURE_MODEL', ''), ('APERTURE_MODEL', 'private token'),
                 ('APERTURE_OLLAMA_HOST', 'file:///private-token'),
                 ('APERTURE_OLLAMA_HOST', 'http://localhost:99999'),
                 ('APERTURE_INFERENCE_TIMEOUT_SECONDS', 'nan'),
                 ('APERTURE_INFERENCE_TIMEOUT_SECONDS', 'inf'),
                 ('APERTURE_INFERENCE_TIMEOUT_SECONDS', '0'),
                 ('APERTURE_INFERENCE_TIMEOUT_SECONDS', '3601'),
                 ('APERTURE_INFERENCE_RETRIES', '-1'), ('APERTURE_INFERENCE_RETRIES', '3'),
                 ('APERTURE_INFERENCE_RETRIES', '1.5'),
                 ('APERTURE_INFERENCE_RETRIES_ACTOR', '1'),
                 ('APERTURE_INFERENCE_NUM_PREDICT', '-1')]
        for name, value in cases:
            with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                with self.assertRaises(config.ConfigurationError) as caught:
                    config.resolve_inference_settings()
                self.assertNotIn('private', str(caught.exception))
        with self.assertRaises(config.ConfigurationError):
            config.resolve_inference_settings('unknown')


class ReliabilityClientTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.client_patch = patch.object(inference.ollama, 'Client')
        self.constructor = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.client = self.constructor.return_value.__enter__.return_value
        self.client.chat.return_value = response()

    def test_resolved_host_timeout_model_and_finite_output_are_passed_to_client(self):
        os.environ.update(APERTURE_OLLAMA_HOST='http://localhost:9876', APERTURE_MODEL='custom:small',
                          APERTURE_INFERENCE_TIMEOUT_SECONDS='2')
        result = inference.chat(role='reflection', messages=[])
        self.assertEqual(result.message.content, 'answer')
        self.constructor.assert_called_once_with(host='http://localhost:9876', timeout=2,
                                                 follow_redirects=False)
        kwargs = self.client.chat.call_args.kwargs
        self.assertEqual(kwargs['model'], 'custom:small')
        self.assertEqual(kwargs['options']['num_predict'], 2048)
        self.assertTrue(self.constructor.return_value.__exit__.called)

    def test_model_options_and_role_override_compatibility(self):
        options = {'temperature': 0, 'num_predict': 64}
        os.environ['APERTURE_MODEL_RETRIEVAL'] = 'retrieval:small'
        inference.role_chat('actor')(role='retrieval', model='explicit:custom', options=options, messages=[])
        self.assertEqual(self.client.chat.call_args.kwargs['model'], 'explicit:custom')
        self.assertEqual(self.client.chat.call_args.kwargs['options'], options)
        self.assertIsNot(self.client.chat.call_args.kwargs['options'], options)

    def test_configured_output_ceiling_bounds_actor_and_retrieval_options(self):
        os.environ.update(APERTURE_INFERENCE_NUM_PREDICT_ACTOR='64',
                          APERTURE_INFERENCE_NUM_PREDICT_RETRIEVAL='32')
        actor_options = dict(llm.ACTOR_OPTIONS)
        inference.chat(role='actor', options=llm.ACTOR_OPTIONS)
        self.assertEqual(self.client.chat.call_args.kwargs['options']['num_predict'], 64)
        self.assertEqual(llm.ACTOR_OPTIONS, actor_options)
        retrieval_options = dict(llm.ACTOR_OPTIONS, temperature=0, seed=42, num_predict=512)
        inference.role_chat('actor')(role='retrieval', options=retrieval_options)
        self.assertEqual(self.client.chat.call_args.kwargs['options']['num_predict'], 32)
        self.assertEqual(self.client.chat.call_args.kwargs['options']['temperature'], 0)
        self.assertEqual(retrieval_options['num_predict'], 512)

    def test_unbounded_or_noninteger_output_limits_rejected_before_client_creation(self):
        for limit in (-1, 0, 2.0, float('inf'), float('nan'), '64', None, True):
            with self.subTest(limit=limit):
                self.constructor.reset_mock()
                with self.assertRaisesRegex(ValueError, 'strictly positive integer'):
                    inference.chat(options={'num_predict': limit})
                self.constructor.assert_not_called()

    def test_each_internal_module_keeps_patchable_call_and_model_symbols(self):
        for module, role in ((reflection, 'reflection'), (self_state, 'self_state'),
                             (relationship, 'relationship'), (task_controller, 'task_controller')):
            with self.subTest(role=role), patch.object(inference, 'chat', return_value=response()) as call:
                result = module.ollama_chat(model=module.MODEL, messages=[])
                self.assertEqual(result.message.content, 'answer')
                self.assertEqual(call.call_args.kwargs['role'], role)
                self.assertEqual(call.call_args.kwargs['model'], module.MODEL)

    def test_actor_never_multiplies_retries(self):
        os.environ['APERTURE_INFERENCE_RETRIES'] = '2'
        self.client.chat.side_effect = ConnectionError('private endpoint detail')
        with self.assertRaises(inference.InferenceError) as caught:
            inference.chat(messages=[])
        self.assertEqual(caught.exception.code, 'connection')
        self.assertEqual(self.client.chat.call_count, 1)
        self.assertNotIn('private', str(caught.exception))

    def test_internal_retries_have_exact_finite_cap(self):
        os.environ['APERTURE_INFERENCE_RETRIES_REFLECTION'] = '2'
        self.client.chat.side_effect = inference.httpx.ReadTimeout('private timeout data')
        with self.assertRaises(inference.InferenceError) as caught:
            inference.chat(role='reflection', messages=[])
        self.assertEqual(caught.exception.code, 'timeout')
        self.assertEqual(self.client.chat.call_count, 3)
        self.assertNotIn('private', str(caught.exception))

    def test_transient_retry_can_recover(self):
        os.environ['APERTURE_INFERENCE_RETRIES_REFLECTION'] = '1'
        self.client.chat.side_effect = [ConnectionError('offline'), response('recovered')]
        self.assertEqual(inference.chat(role='reflection').message.content, 'recovered')
        self.assertEqual(self.client.chat.call_count, 2)

    def test_missing_model_is_explicit_sanitized_and_not_retried(self):
        os.environ['APERTURE_INFERENCE_RETRIES_REFLECTION'] = '2'
        self.client.chat.side_effect = inference.ollama.ResponseError('private model/path', 404)
        with self.assertRaises(inference.InferenceError) as caught:
            inference.chat(role='reflection')
        self.assertEqual(caught.exception.code, 'missing_model')
        self.assertEqual(self.client.chat.call_count, 1)
        self.assertNotIn('private', str(caught.exception))

    def test_server_failure_is_bounded_and_sanitized(self):
        os.environ['APERTURE_INFERENCE_RETRIES_REFLECTION'] = '1'
        self.client.chat.side_effect = inference.ollama.ResponseError('private request contents', 503)
        with self.assertRaises(inference.InferenceError) as caught:
            inference.chat(role='reflection')
        self.assertEqual(caught.exception.code, 'server_error')
        self.assertEqual(self.client.chat.call_count, 2)
        self.assertNotIn('private', str(caught.exception))

    def test_malformed_responses_fail_explicitly_without_retry(self):
        os.environ['APERTURE_INFERENCE_RETRIES_REFLECTION'] = '2'
        for malformed in (None, {}, response(32), SimpleNamespace(message=None), response(None)):
            with self.subTest(malformed=malformed):
                self.client.chat.reset_mock()
                self.client.chat.return_value = malformed
                with self.assertRaises(inference.InferenceError) as caught:
                    inference.chat(role='reflection')
                self.assertEqual(caught.exception.code, 'malformed_response')
                self.assertEqual(self.client.chat.call_count, 1)
        self.client.chat.side_effect = json.JSONDecodeError('private body', '{', 0)
        with self.assertRaises(inference.InferenceError) as caught:
            inference.chat(role='reflection')
        self.assertNotIn('private', str(caught.exception))

    def test_tool_only_and_incomplete_actor_responses_remain_caller_owned(self):
        for reply in (response(None, tool_calls=[]), response('', thinking='reasoning')):
            self.client.chat.return_value = reply
            self.assertIs(inference.chat(), reply)

    def test_malformed_tool_call_shape_is_explicit_failure(self):
        for call in (SimpleNamespace(), SimpleNamespace(function=None),
                     SimpleNamespace(function=SimpleNamespace(name='', arguments={})),
                     SimpleNamespace(function=SimpleNamespace(name='search_memory', arguments='bad'))):
            with self.subTest(call=call):
                self.client.chat.return_value = response('', tool_calls=[call])
                with self.assertRaises(inference.InferenceError) as caught:
                    inference.chat()
                self.assertEqual(caught.exception.code, 'malformed_response')
        valid = SimpleNamespace(function=SimpleNamespace(name='search_memory', arguments={'query': 'synthetic'}))
        self.client.chat.return_value = response(None, tool_calls=[valid])
        self.assertEqual(inference.chat().message.tool_calls, [valid])

    def test_programming_errors_and_interrupts_propagate(self):
        for error in (TypeError('bug'), AttributeError('bug'), RuntimeError('bug'),
                      KeyboardInterrupt(), SystemExit()):
            with self.subTest(error=type(error).__name__):
                self.client.chat.side_effect = error
                with self.assertRaises(type(error)):
                    inference.chat(role='reflection')


class ReflectionFailureTests(unittest.TestCase):
    def test_malformed_model_output_is_failure_before_any_memory_write(self):
        for text in ('', 'not JSON', '{broken', '[]'):
            with self.subTest(text=text), patch.object(reflection, 'ollama_chat', return_value=response(text)), \
                    patch.object(reflection, 'save_reflection_memory') as save:
                with self.assertRaises(inference.InferenceError) as caught:
                    reflection.extract_user_memory('Synthetic user likes ceramic mugs.')
                self.assertEqual(caught.exception.code, 'malformed_response')
                save.assert_not_called()

    def test_legitimate_null_and_schema_rejection_remain_no_memory(self):
        for text in ('null', '{"candidate":null}', '{"candidate":{"category":"invalid","content":"synthetic"}}'):
            with self.subTest(text=text), patch.object(reflection, 'ollama_chat', return_value=response(text)):
                self.assertIsNone(reflection.extract_user_memory('Synthetic user likes ceramic mugs.'))

    def test_failed_write_cannot_return_saved_status(self):
        final = {'user_memory': {'content': 'Synthetic user likes mugs.', 'category': 'preference', 'importance': 3,
                                 'needs_confirmation': False},
                 'self_memory': None}
        with patch.object(reflection, 'REQUIRE_REFLECTION_USER_MEMORY_APPROVAL', False), \
                patch.object(reflection, 'REFLECTION_INTERVAL', 1), \
                patch.object(reflection, '_casual_turns_since_reflection', 0), \
                patch.object(reflection, 'get_recent_observed_message_events', return_value=[]), \
                patch.object(reflection, 'analyze_reflection_debug', return_value={'final': final}), \
                patch.object(reflection, 'save_reflection_memory', side_effect=OSError('synthetic write failed')):
            with self.assertRaises(OSError):
                reflection.maybe_reflect([{'role': 'user', 'content': 'A synthetic conversation about mugs. ' * 5}],
                                         used_action_tool=False, memory_operation_used=False)


if __name__ == '__main__':
    unittest.main()
