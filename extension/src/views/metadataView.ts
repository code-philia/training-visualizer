import * as vscode from 'vscode';
import * as config from '../constants';
import { SidebarWebviewViewProvider } from './sidebarBaseView';

export class MetadataViewManager {
    static provider?: SidebarWebviewViewProvider;

    static get view(): vscode.Webview | undefined {
		return this.provider?.webview;
	}

    private constructor() { }
    
    static getWebViewProvider(): vscode.WebviewViewProvider {
        if (!(this.provider)) {
            this.provider = new SidebarWebviewViewProvider(
                config.isDev ? config.metadataWebviewPort : undefined,
                config.isDev ? '/metadata_view.html' : undefined
            );
        }
        return this.provider;
    }

    // TODO add a base view for metadata view and main plot view
    // cause they coud all postMessage(msg)
    static async postMessage(msg: any): Promise<boolean> {
		if (!(this.view)) {
			return false;
		}
		return await this.view?.postMessage(msg);
	}
}
