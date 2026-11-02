"""Shared test configuration.

`APP_ENV=test` is set before `src.main` is imported, so `.env.test` (which does not exist) is
loaded instead of the developer's `.env.dev`. Default tests never see real secrets.
"""

import os

os.environ["APP_ENV"] = "test"
os.environ.setdefault("ANSWER_PROVIDER", "none")
