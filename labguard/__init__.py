"""LabGuard: grounding natural-language laboratory safety rules into
executable runtime monitors for robotic laboratory manipulation.

Pipeline (Fig. 1):  rule text --grounder--> LabGuard-IR --compiler--> monitors
                    --runtime (SafeFilter + ActionFilter)--> filtered action.
"""

__version__ = "0.1.0"
