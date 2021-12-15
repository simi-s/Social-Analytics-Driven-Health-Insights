from flask import Flask
import pandas as pd
import datetime as dt
from pmaw import PushshiftAPI
api = PushshiftAPI()
from flask import jsonify
from flask import Response


app = Flask(__name__)

def GetIntegerDate(dateString):
    year = int(dateString[6:])
    month = int(dateString[3:5])
    day = int(dateString[:2])
    print('Year is' + str(year) + 'Month is ' + str(month) +  'Day' +str(day))
    return int(dt.datetime(year, month, day, 0, 0).timestamp())

#http://127.0.0.1:5000/fetch/covid19/5000/12-12-2021/12-12-2005
 
@app.route('/fetch/<subreddit>/<limit>/<start_date>/<end_date>')
def main(subreddit,limit,start_date,end_date):
    before = GetIntegerDate(start_date)
    after = GetIntegerDate(end_date)
    print("Subreddit topic is " + subreddit)
    print("Limit is " + limit)
    print("Start Date is " + str(before))
    print("End Date is " + str(after))

    scraped_data = (api.search_comments(
        subreddit=subreddit, limit=int(limit), before=before, after=after))
    df = pd.DataFrame(scraped_data)
    df.info()
    return Response(df.to_json(orient="records"), mimetype='application/json')

if __name__ == "__main__":
    app.run(debug=False,threaded=False)