import * as vscode from 'vscode';
import { getIconUri } from '../resources';
import * as fs from 'fs';
import * as names from '../constants';
import * as api from '../types';
import { isDirectory } from '../utils';

async function repickConfig(configDescription: string, items: (vscode.QuickPickItem & { iconId?: string })[]): Promise<string> {
	const quickPickitems: vscode.QuickPickItem[] = items.map(item => {
		return {
			...item,
			iconPath: item.iconId ? getIconUri(item.iconId) : undefined,
		};
	});
	const picked = await vscode.window.showQuickPick(
		quickPickitems,
		{ placeHolder: configDescription }
	);
	if (!picked) {
		return "";
	}
	return picked.label;
}

function checkDefaultVisualizationConfig(): api.BasicVisualizationOptions | undefined {
	const visConfigSet = vscode.workspace.getConfiguration(names.configurationBaseName);
	const dataType = visConfigSet.get(names.ConfigurationID.dataType);
	const taskType = visConfigSet.get(names.ConfigurationID.taskType);
	const contentPath = visConfigSet.get(names.ConfigurationID.contentPath);
	const visualizationMethod = visConfigSet.get(names.ConfigurationID.visualizationMethod);
	// TODO create a class for a configuration
	// that can both define the ID and validate the value
	if (api.VisualizationTypes.VisualizationDataType.has(dataType) &&
		api.VisualizationTypes.VisualizationTaskType.has(taskType) &&
		typeof contentPath === 'string' && isDirectory(contentPath) && 
		api.VisualizationTypes.VisualizationMethod.has(visualizationMethod)) {
		return {
			dataType: dataType,
			taskType: taskType,
			contentPath: contentPath,
			visualizationMethod: visualizationMethod,
		};
	}
	return undefined;
}

async function reconfigureVisualizationConfig(): Promise<api.BasicVisualizationOptions | undefined> {
	const visConfigSet = vscode.workspace.getConfiguration('timeTravellingVisualizer');	// Should we call this each time?

	const dataType = await repickConfig(
		"Select the type of your data",
		[
			{ iconId: "image-type", label: "Image" },
			{ iconId: "text-type", label: "Text" },
		]
	);
	if (!dataType) {
		return undefined;
	}

	const taskType = await repickConfig(
		"Select the type of your model task",
		[
			{ iconId: "classification-task", label: "Classification" },
			{ iconId: "non-classification-task", label: "Non-Classification" },
		]
	);
	if (!taskType) {
		return undefined;
	}

	// const contentPathConfig = config.get('loadVisualization.contentPath');
	const contentPathConfig = "";
	var contentPath: string = "";
	if (!(typeof contentPathConfig === 'string' && isDirectory(contentPathConfig))) {
		contentPath = await new Promise((resolve, reject) => {
			const inputBox: vscode.InputBox = vscode.window.createInputBox();
			inputBox.prompt = "Please enter the folder path where the visualization should start from";
			inputBox.title = "Data Folder";
			inputBox.placeholder = "Enter the path";
			inputBox.buttons = [
				{ iconPath: vscode.ThemeIcon.Folder, tooltip: "Select folder" }
			];
			const workspacePath = vscode.workspace.workspaceFolders?.[0].uri.fsPath ?? "";
			if (workspacePath) {
				inputBox.value = fs.realpathSync.native(workspacePath);
			}
			inputBox.ignoreFocusOut = true;
			inputBox.valueSelection = [inputBox.value.length, inputBox.value.length];
			function validate(value: string): boolean {
				if (isDirectory(value)) {
					inputBox.validationMessage = "";
					return true;
				} else {
					inputBox.validationMessage = "folder does not exist";
					return false;
				}
			}
			inputBox.onDidTriggerButton(async (button) => {
				if (button.tooltip === "Select folder") {
					const folderPath = await vscode.window.showOpenDialog({
						canSelectFiles: false,
						canSelectFolders: true,
						canSelectMany: false,
						openLabel: "Select folder",
					});
					if (folderPath) {
						const pathResult = folderPath[0].fsPath;
						if (validate(pathResult)) {
							// Don't send it back to input box, because if this uses a simple open dialog, it will immediately close the input box
							resolve(pathResult);
						}
					}
				}
			});
			inputBox.onDidChangeValue((value) => {
				validate(value);
			});
			inputBox.onDidAccept(() => {
				if (validate(inputBox.value)) {
					resolve(inputBox.value);
					inputBox.hide();
				} else {
					inputBox.hide();
					reject("invalid folder path");
				}
			});
			inputBox.onDidHide(() => {
				inputBox.dispose();
			});
			inputBox.show();
		});

		if (!isDirectory(contentPath)) {
			return undefined;
		}
	} else {
		contentPath = contentPathConfig;
	}

	const visualizationMethod = await repickConfig(
		"Select the visualization method",
		[
			{ label: "TrustVis", description: "(default)" }
		]
	);
	if (!visualizationMethod) {
		return undefined;
	}

	visConfigSet.update(names.ConfigurationID.dataType, dataType);
	visConfigSet.update(names.ConfigurationID.taskType, taskType);
	visConfigSet.update(names.ConfigurationID.contentPath, contentPath);
	visConfigSet.update(names.ConfigurationID.visualizationMethod, visualizationMethod);

	return {
		dataType: dataType,
		taskType: taskType,
		contentPath: contentPath,
		visualizationMethod: visualizationMethod,
	};
}

export async function getVisConfig(forceReconfig: boolean = false): Promise<api.BasicVisualizationOptions | undefined> {
	var config: api.BasicVisualizationOptions | undefined;
	if (!forceReconfig) {
		config = checkDefaultVisualizationConfig();
		if (config) {
			return config;
		}
	}
	config = await reconfigureVisualizationConfig();
	return config;
}

export function setDataFolder(file: vscode.Uri | undefined): boolean {
	if (!file) {
		return false;
	}
	const fsPath = file.fsPath;
	if (isDirectory(fsPath)) {
		const config = vscode.workspace.getConfiguration('timeTravellingVisualizer');
		config.update(config.ConfigurationID.contentPath, fsPath);
		return true;
	} else {
		vscode.window.showErrorMessage("Selected path is not a directory 😮");
		return false;
	}
}
