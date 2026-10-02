This is a 60-minute oral "ML design" round with a search specialist. There is no architecture to draw on a whiteboard: every question is about the machine learning underneath one system — how the text embedding model is trained, how retrieval is put together, how results are ranked, and how you would know any of it works.

The system: a document question-answering product built on an internal knowledge base of about 40 million passages (200-400 tokens each, already chunked). About 5 million queries arrive per day, split roughly evenly between short keyword-style queries (2-5 tokens, like a classic search box) and full natural-language questions. The team has a year of click logs (query, the ranked list shown, which result was clicked and at what position) at that volume, plus about 20,000 human-graded `(query, passage, relevance grade 0-3)` judgments, to which an annotation vendor adds a fresh batch every quarter. Target latency for retrieval plus ranking, end to end and excluding any downstream answer generation, is under 150 ms at the median and under 350 ms at the 95th percentile.

Out of scope for this round: the ingestion and chunking pipeline, per-document permissions and multi-tenancy, answer generation and citation checking, and any hardware or capacity estimate. Focus entirely on the models and the retrieval algorithms that turn a query into a ranked list of passages.

Answer the following, moving from training to serving:

### (a) Training the embedding model

- A pretrained text encoder is to be turned into an embedding model for this retrieval task with contrastive learning. Why does that objective fit the task?
- Write out the loss function precisely. What does each term mean, and what is a typical value for the temperature?
- Where would positive pairs come from before any labeled data exists? Where would they come from once labeled data (say, natural-language-inference-style sentence pairs, or question-answer pairs) is available?
- How would negative examples be chosen, and how many per anchor?
- What role does batch size play in training quality? What happens to the loss if you double the batch size, and what changes if you add mined hard negatives on top of the in-batch ones?

### (b) Retrieval flow

- Given a trained embedding model, walk through the path from a raw query to a ranked shortlist of passages.
- Why not score every passage in the corpus against the query with a cross-encoder? What is the division of labor between a bi-encoder and a cross-encoder here?
- What kind of index would you search the embeddings with at this scale, and what is the main knob it exposes?
- Half the query volume behaves like keyword search. Is dense retrieval alone enough for that? How would you combine dense and lexical (sparse) retrieval, and how do you merge two ranked lists whose scores are not on the same scale?
- What is late interaction, as in ColBERT-style retrieval, how does it differ from both the bi-encoder and the cross-encoder, and what does it cost?

### (c) Ranking

- Once hybrid retrieval returns a shortlist, what does a separate ranking stage add? Why not just sort by the retrieval score?
- What kind of model would you use as the reranker, and how does its training data differ from the embedding model's?
- If the best passage for a query never enters the shortlist, can the reranker recover it? What does the answer imply about how you would divide effort between the retrieval stage and the reranker?
- How would you choose the size of the shortlist the reranker scores, and what does that choice trade off?

### (d) Evaluation and iteration

- What offline metrics would you report for the retrieval stage, and what would you report for the final ranked list? Why are they different — think in terms of recall@k, MRR, and nDCG.
- Given the labeled judgments and click logs described above, how would you build an evaluation query set for this product?
- Once this ships, what online signals would you track beyond the offline metrics?
- Months after launch, how would you notice that retrieval quality had quietly gotten worse?
