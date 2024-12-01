import * as vscode from 'vscode';
import { PlotViewManager } from "../views/plotView";
import { getVisConfig, setDataFolder } from './configVisualization';

export async function startVisualization(forceReconfig: boolean = false): Promise<boolean> {
	if (!(PlotViewManager.view)) {
		try {
			await PlotViewManager.showView();
		} catch(e) {
			vscode.window.showErrorMessage(`Cannot start main view: ${e}`);
			return false;
		}
	}
	return await loadVisualizationPlot(forceReconfig);
}

// TODO does passing forceReconfig twice seem silly?
async function loadVisualizationPlot(forceReconfig: boolean = false): Promise<boolean> {
	const config = await getVisConfig(forceReconfig);

	if (config) {
		const { dataType, taskType, contentPath, visualizationMethod } = config;
		return await notifyVisualizationUpdate(dataType, taskType, contentPath, visualizationMethod);
	}
	return false;
}

async function notifyVisualizationUpdate(dataType: string, taskType: string, contentPath: string, visualizationMethod: string): Promise<boolean> {
	const msg = {
		command: 'update',
		contentPath: contentPath,
		customContentPath: '',
		taskType: taskType,
		dataType: dataType,
		forward: true		// recognized by live preview <iframe> (in dev) only
	};
	return await PlotViewManager.postMessage(msg);
}

export function setAsDataFolderAndLoadVisualizationResult(file: vscode.Uri | undefined): void {
    if (!(file instanceof vscode.Uri)) {
        return;
    }	
    const success = setDataFolder(file);
    if (success) {
        startVisualization();
    }	
}
