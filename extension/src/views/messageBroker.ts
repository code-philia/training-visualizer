import * as vscode from 'vscode';
import { PlotViewManager } from "./plotView";
import { MetadataViewManager } from './metadataView';
import { getVisConfig, setDataFolder } from '../processes/configVisualization';

class GlobalMessageBroker {
	static handlers: Map<string, (msg: any) => any> = new Map();
	static defaultHandler: (msg: any) => any = (msg) => {};
	
	static {
		this.initGlobalMessageHandlers();
	}

	static initGlobalMessageHandlers(): void {
		this.addHandler('update', (msg) => {
			// console.log('message: update', msg);
			if (PlotViewManager.view) {
				PlotViewManager.postMessage(msg);
			} else {
				console.log("Cannot find mainView. Message: update not passed...");
			}
		});
		this.addHandler('updateDataPoint', (msg) => {
			// console.log('message: updateDataPoint', msg);
			if (MetadataViewManager.view) {
                msg.command = 'sync';
                // TODO should use MetadataViewManager.view.postMessage
                // for consistency?
				MetadataViewManager.postMessage(msg);
			} else {
				console.log("Cannot find metadata_view");
			}
		});
		this.setDefaultHandler(async (msg) => {
			// In early design, forward it as is to the main view
			// with additional basic configuration fields
			// console.log('message: other type', msg);
			if (PlotViewManager.panel) {
				let config = await getVisConfig();
				if (config) {
					msg.contentPath = config.contentPath;
					msg.customContentPath = '';
					msg.taskType = config.taskType;
					msg.dataType = config.dataType;
					PlotViewManager.panel.webview.postMessage(msg);
				}
			} else {
				console.log("Cannot find mainView. Message: other type not passed...");
			}
		});
	}

	static addHandler(command: string, handler: (msg: any) => void): void {
		this.handlers.set(command, handler);
	}
	
	static setDefaultHandler(handler: (msg: any) => void): void {
		this.defaultHandler = handler;
	}

	// NOTE don't use this static method as an outside handler directly
	// cause `this` is not bound
	static handleMessage(msg: any): boolean {
		// Returns true if there is a command handler for it
		const handler = this.handlers.get(msg.command);
		if (handler) {
			handler(msg);
			return true;
		} else {
			this.defaultHandler(msg);
			return false;
		}
	}

	private constructor() { }
}

export function handleMessageDefault(msg: any): boolean {
	return GlobalMessageBroker.handleMessage(msg);
}
