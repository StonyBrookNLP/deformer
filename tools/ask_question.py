#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests

url = 'http://xxx.local:0000/qa'
for i in range(1):
    questions = ["Which NFL team represented the AFC at Super Bowl 50?",
                 "Which NFL team represented the NFC at Super Bowl 50?"]
    context = "Super Bowl 50 was an American football game to determine the champion of " \
              "the National Football League (NFL) for the 2015 season. The American " \
              "Football Conference (AFC) champion Denver Broncos defeated " \
              "the National Football Conference (NFC) champion " \
              "Carolina Panthers 24\u201310 to earn their third Super Bowl title. " \
              "The game was played on February 7, 2016, at Levi's Stadium in the " \
              "San Francisco Bay Area at Santa Clara, California. As this was the" \
              " 50th Super Bowl, the league emphasized the \"golden anniversary\" with " \
              "various gold-themed initiatives, as well as temporarily suspending the " \
              "tradition of naming each Super Bowl game with Roman numerals (under which " \
              "the game would have been known as \"Super Bowl L\"), so that the logo could " \
              "prominently feature the Arabic numerals 50."
    contexts = [context, context]
    data = {"question": questions, "context": contexts}
    response = requests.post(url, json=data)
    response_content = response.json()
    answers = response_content['answer']
    scores = response_content['score']
    for q, c, a, s in zip(questions, contexts, answers, scores):
        print('q={}\na={}\n\tcontext={}\n\n'.format(q, (a, s), c))

"""
export POST_URL='http://xxx.local:0000/qa'
curl --header "Content-Type: application/json" \
  --request POST $POST_URL \
  --data @- << EOF
{
"question":"Brunch time on Sunday",
"context":"hours of operation. brunch: saturday: 10:30am - 3:00pm, \
sunday: 10:30am - 3:00pm; dinner: monday - friday: 5:00pm - 11:00pm; \
saturday - sundayr: 3:00pm - 11:00pm."
}
EOF

"""
