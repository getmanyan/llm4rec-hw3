# Collaborative Filtering Movie Recommender (HW3, LLM4Rec)

Live page: https://getmanyan.github.io/llm4rec-hw3/

Pick a MovieLens user and get two Top-5 lists side by side:

- **User-Based CF**: the 20 users with the most similar ratings; score = their similarity-weighted average rating (1-5). A movie needs at least 3 of the 20 neighbours to have rated it.
- **Item-Based CF**: movie-to-movie similarity of rating columns; score = similarity to the user's rated movies, weighted by the user's ratings (0-1).

There is also a "Predict one rating" block (user + movie -> predicted rating with both approaches), as on the lecture slide.

**Missing values:** weighted approach. Cosine similarity over co-rated entries only, multiplied by `min(n, 50) / 50`, where `n` is the number of co-rated entries.

**Data fixes in `data.js`** (dataset files are unchanged): 19 genre flags instead of 18, Latin-1 decoding of `u.item`, 18 duplicate titles merged (1682 ids -> 1664 movies).

## Files

| File | What it does |
|---|---|
| `index.html`, `style.css` | page |
| `data.js` | loading, parsing, duplicate merge, rating matrix |
| `script.js` | similarity, User-Based and Item-Based CF, rendering |
| `analysis/harness.js` | loads the app's own `data.js` + `script.js` in Node |
| `analysis/evaluate.js` | offline experiments for the report |
| `analysis/results.txt` | output of the last run |
| `assignment_prompt.md` | the original week3 specification |

## Reproduce the numbers

```
node analysis/evaluate.js
```

Node 18+, no dependencies, about 5 minutes. The split is seeded, so the quality numbers repeat exactly; timings differ by machine.

Starter code: [dryjins/RecSys-LLMs/week3](https://github.com/dryjins/RecSys-LLMs/tree/main/week3). Data: MovieLens 100K (GroupLens).
