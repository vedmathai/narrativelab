# Narrative Lab

The aim of this project is to further foundational research to detect and track narratives. It is run between the University of Oxford, The Alan Turing Institute, and Defense Science and Technology Laboratory (dstl).

# Running the Narrative Graph Exploration locally
You will need to run
1. Python Server
2. React UI

## Creating Working Directory
Though users are free to create their own form of working directories, later steps depend on the code being run from the correct level in the working directory tree. To make the later instructions easy to follow, follow the steps to create the same working directory path.

```
cd <PATH/TO/FOLDER>
mkdir narrative_project
cd narrative_project
```
## Clone the repository
`git clone https://github.com/MaximilianAhrens/narrativelab.git`

## Create the Virtual Ennvironment
`python3 -m venv venv_narrative_project`

## Activate the Virtual Environemnt
You will have to do this each time you open a new terminal session or tab
`source venv_narrative_project/bin/activate`
Windows: `venv_narrative_project\Scripts\activate`

## Enter the git project folder
`cd narrativelab`

## Install Python Requirements
`pip install -r narrativity/requirements.txt`

## Install Spacy Dependencies
RUN `python -m spacy download en_core_web_lg` to install the spacy models.
RUN `python -m coreferee install en` to install the coreferee (coreference resolution) models.

## Run Python Server
`PYTHONPATH=. python narrativity/server/run.py` 
Windows: `$env:PYTHONPATH='.'; python .\narrativity\server\run.py`

Adding the `PYTHONPATH` before the code invocation is important because the python code treats itself like a package called `narrativity`.
Alternatively one could add this folder permanently to their `PYTHONPATH`, but may come with its own overhead.

If the installation went well, you should see the server turn on with the following message:
```
 * Serving Flask app 'run'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 122-965-760
 ``` 

 ## Test that the NLP pipeline is working and all dependencies have been installed
 Run `PYTHONPATH=. python narrativity/graph_generator/tests/graph_generator_parser_test.py`
 Windows: `$env:PYTHONPATH='.'; python narrativity/graph_generator/tests/graph_generator_parser_test.py`

 The code will provide a JSON of the narrative graph.
 
 ## Using the narrative graph in downstream applications
 The test file used in the previous step `narrativity/graph_generator/tests/graph_generator_parser_test.py` shows and example of how to use the graph in downstream applications.

    When running the downstream application, make sure the narrativity package is in your `PYTHONPATH`.
    ```
    from narrativity.graph_generator.dependency_parse_pipeline.parser import NarrativeGraphGenerator
    ngg = NarrativeGraphGenerator()
    ngg.load()
    narrative_graph = ngg.generate(text)
    ```

The `ngg.generate(text)` returns an object which is an instance of the `NarrativeGraph` class. The defintion of this class is at `narrativity/datamodel/narrative_graph/narrative_graph.py`.

`narrative.to_dict()` will serialize the narrative to a dictionary. Use the serialization only if there is a need to print the information or download it to file, in order to move it around or save it. In most other cases it would be better to deal with the Python objects themselves directly.

To deserialize use
```
import json

from narrativity.datamodel.narrative_graph.narrative_graph import NarrativeGraph

with open('serialized_narrative_graph.json') as f:
    narrative_graph = NarrativeGraph.from_dict(json.load(f))

```

# Installing the React User Interface requirements and running it

## Install npm
You will need npm on your local machine to go forward.
For Ubuntu see [here](https://linuxize.com/post/how-to-install-node-js-on-ubuntu-20-04/?utm_content=cmp-true).
For Mac see [here](https://treehouse.github.io/installation-guides/mac/node-mac.html).
For Windows see [here](https://phoenixnap.com/kb/install-node-js-npm-on-windows).


## Go to UI src folder.
1. Open a new tab in your terminal.
2. Activate the same venv. (Not entirely sure if React follows the Python venv, but it feels like it does.)
3. Assuming your current directory is the `narrativelab` folder, RUN `cd client/narrativity-app`

## Install npm dependencies
RUN `npm i`

## Start the server
RUN `npm start`

A new browser should open up pointed at `http://localhost:3000`. You can use a browser of your choice and point at this same localhost URL.

# Using the User Inferface.
1. Enter a sample corpus in the text box and click the `Submit` button.
2. Click on a narrative node card to move to the `narrative-context` view for that node.
3. Click once on a card that is not the `node-under-observation` to see the relationship information for the relationship between the `node-under-observation` and that node.
4. Double-click on a card that is not the `node-under-observation` to go to the view of that card.
5. Flip between `Node Context` and `Discover` tabs to search for more nodes.
6. Search and Filter does NOT currently work.
7. Move back to the `HOME` tab to run another sentence.
8. NOTE these are early versions and therefore may not be stable.



# Git Guideline
0. Create a new branch if neccessary with `git checkout -b myNewBranchName`
1. Git pull to be up to date with : `git pull --rebase orgin main`
2. Git add all files with : `git add .`
3. Git commit any changes made with : `git commit -m 'Text of my changes'`
4. Git push to origin with : `git push origin HEAD`

5. wait for review changes & review those of your team-members 

6. Merge pull request once the reviews are accepted with the button on Github
7. Pull the accepted merged changes to your local git with : `git pull --rebase origin main`

Note if you have temporary changes you don't want cloberred, commit to temp with `git commit -m 'temp'`
Then : `git pull --rebase origin main`
Then `git reset --soft HEAD^` to reset local changes to uncommited 


