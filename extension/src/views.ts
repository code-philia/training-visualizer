import * as vscode from 'vscode';
import * as constants from './constants';
import { MetadataViewManager } from './views/metadataView';

export function doViewsRegistration(): vscode.Disposable {
    const metadataViewRegistration = vscode.window.registerWebviewViewProvider(
        constants.ViewsID.metadataView,
        MetadataViewManager.getWebViewProvider(),
        { webviewOptions: { retainContextWhenHidden: true } }
    );
    return vscode.Disposable.from(metadataViewRegistration);
}
