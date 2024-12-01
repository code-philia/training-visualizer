import * as vscode from 'vscode';
import * as config from '../constants';
import { getLiveWebviewHtml } from '../dev/devServer';
import { handleMessageDefault } from './messageBroker';


export class SidebarWebviewViewProvider implements vscode.WebviewViewProvider {
	private readonly port?: number;
	private readonly path?: string;
	public webview?: vscode.Webview;

	constructor(port?: number, path?: string) {
		this.port = port;
		this.path = path;
	}

	public resolveWebviewView(
		webviewView: vscode.WebviewView,
		context: vscode.WebviewViewResolveContext,
		token: vscode.CancellationToken
	) {
		this.webview = webviewView.webview;

		webviewView.webview.options = config.getDefaultWebviewOptions();

		if (!this.port) {
			webviewView.webview.html = this.getPlacehoderHtmlForWebview(webviewView.webview);		// FIXME: the static version (for prod) is not updated																					// to be the same as live preview version (for dev) yet
			return;
		}

		webviewView.webview.html = getLiveWebviewHtml(webviewView.webview, this.port, false, this.path);
		webviewView.webview.options = {
			enableScripts: true,
		};

		webviewView.webview.onDidReceiveMessage(handleMessageDefault);
	}

	private getPlacehoderHtmlForWebview(webview: vscode.Webview): string {
		return `<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>My Custom View</title>
        </head>
        <body>
            <h1>Hello from My Custom View!</h1>
        </body>
        </html>`;
	}
}
