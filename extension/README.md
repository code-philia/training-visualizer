# <img src="resources/vscode.ico" alt="vscode-logo" style="height:1.2em; vertical-align:text-bottom;"> VS Code Extension for Time Travelling Visualizer

## Features

> 🏗️ The extension is still in construction...

Run `Visualize Training Process` or with config `Configure and Visualize Training Process` command to start visulizing training process for a selected data folder.

## Development Guide: To be short

We are currently on `feat/vscode-extension` branch for visualizer extension development, where the source code is in the `extension` folder. Install the dependencies:

```
npm install
```

Then in VS Code debugger, use `Launch Visualizer Backend` option to launch backend, after which use `Run Extension` to debug the frontend.

## Development Guide: For beginners

> We would have assumed experienced front-end developers with decent VS Code knowledge to join this project, but usally young contributors are invited.
> Thus, a more comprehensive guide is written to lead junior developers before they could be pro.

The following guide pre-assumes :

1. Basic programming and environment setting up experience
2. Knowing how to use Git, VS Code and Node.js

### Tools

+ **Terminal:** Any pre-installed terminals and shells on your system should probably be compatible.
+ **VS Code:** Go to [VS Code official website](https://code.visualstudio.com/) and download the latest version of Visual Studio Code.
+ **Git:** [Git Downloads](https://git-scm.com/downloads)
+ **Node.js:** [Node.js official website](https://nodejs.org/)

> After installing Git and Node.js, if you use *VS Code terminal on Windows*, rememeber to close all VS Code windows (not just reloading them) and open them again to make sure new PATH environment variables are loaded.

### Get the project

Using HTTPS to clone project is more convenient. Run this in the terminal:

```shell
git clone https://github.com/code-philia/time-travelling-visualizer.git
```

### Open it using VS Code

Open the project folder `time-travelling-visualizer` **as the workspace folder** in VS Code. Three ways to do this:

+ Open VS Code, drag the project folder into the window or choose `File -> Open Folder...` in the menu
+ Open the terminal, `cd` to the project folder, and run `code .`
+ Right click the project folder and choose `Open in VS Code`, only if you have enable the context menu option when installing on Windows

> Some advanced users may open their whole personal folder for development as the workspace folder, but on development of this project you would more like to set the `time-travelling-visualizer` project folder as the workspace folder, so that the `.vscode` and `.git` directories in the project could be recognized by VS Code. If not, please merge `launch.json` and `tasks.json` into your workspace folder's `.vscode` directory change to paths in them to the relative path from your workspace folder.
>
> ```json
> // launch.json
> "args": [
>    "--extensionDevelopmentPath=<your-own-relative-path-to-extension-folder>",
>    "--disable-extensions",
> ],
> "outFiles": [
>    "<your-own-relative-path-to-extension-folder>/out/**/*.js"
> ],...
>
> // tasks.json
> "path": "<your-own-relative-path-to-extension-folder>",...
> ```

### Switch to extension development branch

We suggest to use the terminal embedded in VS Code, which is launched from your workspace folder. `Ctrl+~` to open the terminal.

```shell
git checkout feat/vscode-extension
```

To confirm your success, make sure `<project-folder>/.vscode` contains `launch.json` and `tasks.json` files, and `<project-folder>/extension` contains a VS Code extension with `src` folder, `package.json` and other configuration files.

### Install dependencies

In the terminal, go to the `extension` folder install Node.js modules:

```shell
cd extension
npm install
```

### Run the extension

In the *Run and Debug* tab <img src="resources/run-and-debug.png" alt="run-and-debug" style="height: 2em; vertical-align: middle;"> on the left activity bar of vscode, choose *Run Extension* option of launch <img src="resources/run-extension.png" alt="run-extension" style="height: 3em; vertical-align: middle;"> , then press the button to launch an Extension Host window.

If succeeded, you should

+ see a new view container in the *left activity bar* and a new view container in the *lower panel* of VS Code. If not seeing the panel, toggle it <img src="resources/toggle-panel.png" alt="toggle-panel" style="height: 2em; vertical-align: middle;"> on the top left of the window.
+ `Ctrl+Shift+P` to search for the command "Visualize Training Process" and execute it to *specify a data folder and see a new editor for visualization.*

## Known Issues

> 🏗️ The extension is still in construction...

## Release Notes

> 🏗️ The extension is still in construction...
