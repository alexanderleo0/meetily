"""Checks for benchmark accounting and safe report rendering."""
import json
from pathlib import Path
import tempfile
import unittest
import compare


class ComparisonTests(unittest.TestCase):
    def test_normalization_keeps_digits_and_words(self):
        self.assertEqual(compare.normalized('Ёж, ПВЗ! 2026 — тест.'), 'еж пвз 2026 тест')

    def test_report_escapes_transcripts(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            result = dict(model='test', inference_s=1, rtfx_speech=2, rtf_speech=.5,
                p95_chunk_s=.2, chunks_slower_than_realtime=0, peak_rss_mib=50,
                segments=[dict(start_s=1, end_s=2, text='<script>bad()</script>', inference_s=.1)])
            for name in ['parakeet', 'gigaam']:
                (out / f'{name}.json').write_text(json.dumps(result))
            compare.report(out, ['parakeet', 'gigaam'])
            page = (out / 'report.html').read_text()
            self.assertIn('&lt;script&gt;bad()', page)
            self.assertNotIn('<script>bad()', page)
            summary = json.loads((out / 'summary.json').read_text())
            self.assertNotIn('segments', summary['parakeet'])


if __name__ == '__main__':
    unittest.main()
