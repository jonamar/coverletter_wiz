#!/usr/bin/env python3
"""
DEPRECATED: JobAnalyzer functionality has been migrated to the report command.

This file is kept for historical reference but the tests are no longer valid
since the standalone JobAnalyzer functionality has been consolidated into the report command.

Please refer to the following test files for testing the new integrated functionality:
- test_html_utils.py: Tests for HTML content extraction
- test_report_job_analysis.py: Tests for job analysis functionality in the report command
- test_report_url_integration.py: Tests for report generation with job URLs
"""

import unittest
from typing import Optional


class TestJobAnalyzerDeprecated(unittest.TestCase):
    """DEPRECATED: JobAnalyzer tests have been moved.
    
    This test class is kept for historical reference but is disabled since the
    functionality has been migrated to the report command.
    
    The JobAnalyzer functionality has been consolidated into the report command
    as part of a code cleanup to reduce duplication and improve maintainability.
    """
    
    def test_deprecated_notice(self) -> None:
        """Test to remind that these tests are deprecated.
        
        This test doesn't actually test any functionality, but serves as a
        reminder that the JobAnalyzer tests have been migrated to other test files.
        
        Returns:
            None
        """
        print("NOTICE: JobAnalyzer functionality has been migrated to the report command.")
        print("Please refer to test_html_utils.py and test_report_job_analysis.py for the new tests.")
        # This test passes to avoid test failures, but outputs a notice
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
