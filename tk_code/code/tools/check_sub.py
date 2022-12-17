import os
import time
import datetime
from datetime import timezone
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

COMPETITION = 'feedback-prize-english-language-learning'
result = api.competition_submissions(COMPETITION)[0]
latest_ref = str(result)  # latest submission
submit_time = result.date
notebook_url = result.url
notebook_name = result.fileName
notebook_description = result.description
notebook_error = result.errorDescription

status = ''

while status != 'complete':

    list_of_submission = api.competition_submissions(COMPETITION)
    for result in list_of_submission:
        if str(result.ref) == latest_ref:
            break
    status = result.status

    now = datetime.datetime.now(timezone.utc).replace(tzinfo=None)
    elapsed_time = int((now - submit_time).seconds / 60) + 1
    if status == 'complete':
        message = f'{{"text": "■Notebook name: {notebook_name}\n■Notebook url: {notebook_url}\n■Notebook description: {notebook_description}\n■Time: {elapsed_time} min\n■Public LB: {result.publicScore}\n■Notebook error: {notebook_error}"}}'
        print(message)
    else:
        time.sleep(60)