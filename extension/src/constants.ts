// NOTE only and always import this ES module and api.ts as an entire namespace:
// `import * as constants from './constants';`
// and always import other modules by each symbol:
// `import {xxx} from './yyy';`

import * as vscode from 'vscode';
import * as path from 'path';

export let isDev = true;
export const editorWebviewPort = 5001;
export const controlWebviewPort = 5002;
export const metadataWebviewPort = 5003;

export class GlobalStorageContext {
	private static extensionLocation: string = __dirname;
	private static readonly webRootRelativePath = 'web/';
	private static readonly resourceRootRelativePath = 'resources/';
	
	private constructor() { }

	static get webRoot(): string {
		return path.join(this.extensionLocation, this.webRootRelativePath);
	}

	static get resouceRoot(): string {
		return path.join(this.extensionLocation, this.resourceRootRelativePath);
	}

	static initExtensionLocation(root: string): void {
		this.extensionLocation = root;
	}
}

// TODO this should not be put in this module?
export function getDefaultWebviewOptions(): vscode.WebviewOptions {
	const resourceUri = vscode.Uri.file(GlobalStorageContext.webRoot);
	// console.log(`Resource URI: ${resourceUri}`);
	return {
		"enableScripts": true,
		"localResourceRoots": [
			resourceUri
		]
	};
}

function withBaseName(id: string): string {
    return `${configurationBaseName}.${id}`;
}

export const configurationBaseName = 'timeTravellingVisualizer';

export class ConfigurationID {
    static readonly dataType = 'loadVisualization.dataType';
    static readonly taskType = 'loadVisualization.taskType';
    static readonly contentPath = 'loadVisualization.contentPath';
    static readonly visualizationMethod = 'loadVisualization.visualizationMethod';

    private constructor() {}
}

export class CommandID {
    static readonly loadVisualization = withBaseName('loadVisualizationResult');
    static readonly openStartView = withBaseName('start');
    static readonly setAsDataFolderAndLoadVisualizationResult = withBaseName('setAsDataFolderAndLoadVisualizationResult');
    static readonly setAsDataFolder = withBaseName('setAsDataFolder');
    static readonly configureAndLoadVisualization = withBaseName('configureAndLoadVisualizationResult');
}

export class ViewsID {
	static readonly metadataView = 'visualizer-metadata-view';
}
