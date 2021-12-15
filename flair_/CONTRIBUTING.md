# Contributing to Flair

We are happy to accept your contributions to make `flair` better and more awesome! To avoid unnecessary work on either 
side, please stick to the following process:

1. Check if there is already [an issue](https://github.com/zalandoresearch/flair/issues) for your concern.
2. If there is not, open a new one to start a discussion. We hate to close finished PRs!
3. If we decide your concern needs code changes, we would be happy to accept a pull request. Please consider the 
commit guidelines below.

In case you just want to help out and don't know where to start, 
[issues with "help wanted" label](https://github.com/zalandoresearch/flair/labels/help%20wanted) are good for 
first-time contributors. 


## Git Commit Guidelines

If there is already a ticket, use this number at the start of your commit message. 
Use meaningful commit messages that described what you did.

**Example:** `GH-42: Added new type of embeddings: DocumentEmbedding.` 


## Running unit tests locally

For contributors looking to get deeper into the API we suggest cloning the repository and checking out the unit
tests for examples of how to call methods. Nearly all classes and methods are documented, so finding your way around
the code should hopefully be easy.

You need [Pipenv](https://pipenv.readthedocs.io/) for this:

```bash
pipenv install --dev && pipenv shell
pytest tests/
```

To run integration tests execute:
```bash
pytest --runintegration tests/
```
The integration tests will train small models.
Afterwards, the trained model will be loaded for prediction.

To also run slow tests, such as loading and using the embeddings provided by flair, you should execute:
```bash
pytest --runslow tests/
```
