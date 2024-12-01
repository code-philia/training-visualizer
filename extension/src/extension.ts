import * as vscode from 'vscode';
import * as constants from './constants';
import { startDefaultDevLiveServers } from './dev/devServer';
import { doCommandsRegistration } from './commands';
import { doViewsRegistration } from './views';

export function activate(context: vscode.ExtensionContext): void {
	// Cannot read args in launch.json due to
	// vscode using an extension host to manage extensions
	// Setting isDev directly here
	constants.GlobalStorageContext.initExtensionLocation(context.extensionUri.fsPath);

	if (constants.isDev) {
		console.log(`Enabling dev mode locally. Webviews are using live updated elements...`);
		startDefaultDevLiveServers(context);
	}

	context.subscriptions.push(doCommandsRegistration());
	context.subscriptions.push(doViewsRegistration());
}

export function deactivate(): void { }
