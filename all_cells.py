"""Legacy compatibility wrapper.

The project has been refactored into the `field_planner` package.
Use `python -m field_planner.run --demo` or `streamlit run app.py` for the current workflow.
"""

from field_planner.run import main


if __name__ == "__main__":
    main()
