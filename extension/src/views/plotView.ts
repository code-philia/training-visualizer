import * as vscode from 'vscode';
import * as path from 'path';
import * as config from '../constants';
import { readFileSync } from 'fs';
import { handleMessageDefault } from './messageBroker';
import { getLiveWebviewHtml } from '../dev/devServer';

function replaceUri(html: string, webview: vscode.Webview, srcPattern: string, dst: string): string {
	// replace all 'matched pattern' URI using webview.asWebviewUri,
	// which is hosted by VS Code client,
	// or it cannot be loaded
	// where the regex pattern should yield the first group as a correct relative path
	const cssFormattedHtml = html.replace(new RegExp(`(?<=href\="|src\=")${srcPattern}(?=")`, 'g'), (match, ...args) => {
		if (match) {
			// console.log(`matched: ${match}`);
			const formattedCss = webview.asWebviewUri(vscode.Uri.file(path.join(dst, args[0])));
			return formattedCss.toString();
		}
		return "";
	});

	return cssFormattedHtml;
}

function loadHomePage(webview: vscode.Webview, root: string, mapSrc: string, mapDst: string): string {
	const html = readFileSync(root, 'utf8');
	return replaceUri(html, webview, mapSrc, mapDst);
}

export class PlotViewManager {
	static panel?: vscode.WebviewPanel;
	
	static get view(): vscode.Webview | undefined {
		return this.panel?.webview;
	}

	private constructor() { }

	static async createPanel(): Promise<boolean> {
		const panel = vscode.window.createWebviewPanel(
			'plotView',
			'Visualizer',
			vscode.ViewColumn.One,
			{ retainContextWhenHidden: true, ...config.getDefaultWebviewOptions() }
		);
	
		panel.iconPath = vscode.Uri.file(path.join(config.GlobalStorageContext.webRoot, '..', 'resources/eye_tracking_24dp_5F6368_FILL0_wght300_GRAD0_opsz24.svg'));
	
		if (config.isDev) {
			panel.webview.html = getLiveWebviewHtml(panel.webview, config.editorWebviewPort, true);
		} else {
			panel.webview.html = loadHomePage(
				panel.webview,
				path.join(config.GlobalStorageContext.webRoot, 'index.html'),
				'(?!http:\\/\\/|https:\\/\\/)([^"]*\\.[^"]+)',	// remember to double-back-slash here
				config.GlobalStorageContext.webRoot
			);
		}
		
		panel.onDidChangeViewState((e) => {
			console.log(`Panel view state changed: ${e.webviewPanel.active}`);
		});
		panel.onDidDispose((e) => {
			this.panel = undefined;
		});
	
		// TODO the iframe would not be refreshed for not receiving "update" message, which is a handicap for live preview in development
		// reload the data when the iframe is refreshed, maybe by posting a message to vscode to ask for several major arguments
		panel.webview.onDidReceiveMessage(handleMessageDefault);
		
		const loaded: Promise<boolean> = new Promise((resolve) => {
			panel.webview.onDidReceiveMessage((msg) => {		// this will add a listener, not overwriting
				if (msg.state === 'load') {
					this.panel = panel;
					resolve(true);
				}
			});
		});
		return loaded;
	}

	static async postMessage(msg: any): Promise<boolean> {
		if (!(this.view)) {
			return false;
		}
		return await this.view?.postMessage(msg);
	}

	static async showView(): Promise<boolean> {
		if (!(PlotViewManager.panel)) {
			await PlotViewManager.createPanel();
			return true;
		}
		// TODO handle error and return false, not always return true
		return true;
	}
}
