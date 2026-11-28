"""Publication checks use synthetic values, never real credentials or private links."""

import io
import unittest
import zipfile

from scripts.check_publication import content_issues, file_issues


class PublicationChecks(unittest.TestCase):
    def test_reviewed_public_links_and_sponsorships(self):
        for value in (
            b"https://github.com/Exciton-Research/robogogogo.git",
            b"git@github.com:Exciton-Research/RoboGoGoGo.git",
            b"https://github.com/google-deepmind/mujoco_menagerie/tree/main",
            b"https://github.com/sponsors/ai",
        ):
            with self.subTest(value=value):
                self.assertEqual(content_issues(value), [])

    def test_unreviewed_repository_links(self):
        for prefix in (b"https://github.com/", b"git@github.com:"):
            value = prefix + b"example/" + b"unreviewed"
            self.assertEqual(content_issues(value), ["unreviewed GitHub repository link"])

    def test_personal_paths(self):
        for value in (
            b"/home/" + b"someone/project",
            b"/Users/" + b"someone/project",
            b"C:\\Users\\" + b"someone\\project",
        ):
            self.assertEqual(content_issues(value), ["personal absolute path"])

    def test_credentials(self):
        for value in (b"AKIA" + b"A" * 16, b"ghp_" + b"a" * 36):
            self.assertEqual(content_issues(value), ["credential"])
        value = b"-----BEGIN " + b"PRIVATE KEY-----"
        self.assertEqual(content_issues(value), ["private key"])

    def test_embedded_archive_and_redacted_output(self):
        stream = io.BytesIO()
        value = b"https://github.com/" + b"example/" + b"unreviewed"
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("metadata.txt", value)
        issues = file_issues("asset.zip", stream.getvalue())
        self.assertTrue(issues)
        self.assertNotIn(value.decode(), "\n".join(issues))
        self.assertIn("embedded archive content", "\n".join(issues))


if __name__ == "__main__":
    unittest.main()
