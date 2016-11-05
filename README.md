# A simple game recommendation app.

A simple game recommendation engine using data scraped from BoardGameGeek. Given ratings for a few games or a username, it will return ten games rated by their similarity to your ratings.

Also, if you know the properties of game you'd like to make, we can show you the predicted BoardGameGeek score and the most similar games.

Currently, the ratings have a slight bias towards wargames. This skewing will be removed in future iterations.

## Features to come:
- Replace the similarity based approach with a matrix factorization approach
- Use a better model to predict rating from game properties
- Clean up my code and add the scripts I used to generate the matrices

I would also like to build a database of board game manuals, and use a deep learning approach to generate new manuals for new games.

## Files:
- app.py is the main flask app.
- gameData.p is a pickle of game properties and descriptions.
- aMatrixHalfFloat and aMatrixMaskedHalfFloat are two pickled matrices of essentially sums of the cosine similarities of roughly half a million users stored as half precision floats.
- gamegameHalfFloat is a pickled matrix of the Tanimoto similarities of normalized game properties.
- itemitemHalfFloat is a pickled matrix of the Tanimoto similarities of game ratings from users.
- gamescoremodel is a pickled random forest related game properties to their BoardGameGeek score.
- static is a folder of static assets.
- templates is a folder of templates for pages. 