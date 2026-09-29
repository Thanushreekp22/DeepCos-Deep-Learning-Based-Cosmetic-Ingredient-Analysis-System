"""
DeepCos rule layer: knowledge-base driven ingredient screening and reporting.

profile         : score -> band mapping and profile summaries
concern_engine  : potential-concern screening over documented ingredient facts
report_builder  : renders a DeepCos report dict into Markdown / plain text
"""

from ml.rules import concern_engine, profile, report_builder

__all__ = ["concern_engine", "profile", "report_builder"]
