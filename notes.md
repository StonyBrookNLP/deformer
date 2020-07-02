
## Pairwise Dataset Unification

Each data instance has four fields: `id`, `seq1`, `seq2` and `label`.
The `label` field has two children fields: `cls` and `ans`, 
where `cls` indicates the dataset specific class(e.g. yes or no answer for QA),
and `ans` field provides answer span information (answer text and its 
character offset in the context). 

````json
{
  "id": "unique id",
  "seq1": "first text sequence",
  "seq2": "second text sequence",
  "label": {
    "cls": 0,
    "ans": [
      [
        0,
        "text"
      ]
    ]
  }
}

````

### Extractive QA (SQuAD 1.1, SQuAD 2.0 and HotpotQA)

- `seq1` can be the question text and `seq2` can be the paragraph context.

- `cls` in SQuAD 1.1 is optional (`0` means span answer) and `cls` in SQuAD 
2.0 can have two values: `0`(span answer) and `1`(no answer). 
HotpotQA can set `cls` as three values:  `0`(span answer), `1`(_yes_ answer), 
and `2`(_no_ answer). 

- `ans` is a list of pairs (firs one is answer offset and second one is
 answer string) for all three datasets.


Example:

````json
{
  "id": "492c165",
  "seq1": "In what country is Normandy located?",
  "seq2": "The Normans were the people who gave their name to Normandy, a region in France.",
  "label": {
    "cls": 0,
    "ans": [
      [
        73,
        "France"
      ]
    ]
  }
}
````

### BoolQ, MultiNLI and QQP

- `seq1` and `seq2` are the given paired input texts.

- `cls` is the class label

````json
{
  "id": "63735n",
  "seq1": "The new rights are nice enough",
  "seq2": "Everyone really likes the newest benefits ",
  "label": {
    "cls": 2
  }
}
````

### Semantic Textual Similarity

- `cls` is the similarity score (float number)

## Dataset Preprocessing

cleanup raw text -> tokenization -> adjust labels

map ids to prediction text

