import unittest

from preprocessing import infer_session_code


class SessionCodeInferenceTest(unittest.TestCase):
    def test_identifies_fp2(self):
        session = type("Session", (), {"name": "Practice 2"})()
        self.assertEqual(infer_session_code(session), "FP2")

    def test_identifies_sprint(self):
        session = type("Session", (), {"session_info": {"Name": "Sprint"}})()
        self.assertEqual(infer_session_code(session), "S")

    def test_defaults_to_race(self):
        session = type("Session", (), {"name": "Race"})()
        self.assertEqual(infer_session_code(session), "R")
