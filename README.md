# Text Summarization

This repository contains implementation of a text-summarization model and a keyword-extraction model. This is a university course project in the field if Information Retrieval. The models implemented follow closely the papers:
 * *TextRank: Bringing Order into Texts* by Rada Mihalcea and Paul Tarau
 * *Graph-based Ranking Algorithms for Sentence Extraction, Applied to Text Summarization* by Rada Mihalcea

The following models are implemented:
 * `text_summarizer.py` implements a text-summarization model that uses Google PageRank algorithm to rank sentences in a text. Every sentence is added as a vertex in a weighted graph. Edges between vertices are weighed according to the similarity score between sentences. Similarity between two sentences is measured as a function of their content overlap.
 * `keyword_extractor.py` implements a keyword-extraction model that uses Google PageRank algorithm to rank words in a text. Every word is added as a vertex in an unweighted graph. An edge between two vertices is added if the corresponding words co-occur within a window of a given size.

 After the graph is constructed the PageRank algorithm is run on the graph to compute the score associated with each vertex. Once a final score is obtained for each vertex in the graph, vertices are sorted in reversed order of their score, and the top *K* vertices in the ranking are retained for post-processing.