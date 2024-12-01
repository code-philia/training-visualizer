import * as vscode from 'vscode';
import * as config from './constants';

export const iconPaths: {[key: string]: string} = {
    "image-type": "imagesmode_24dp_5F6368_FILL0_wght400_GRAD0_opsz24.svg",
    "text-type": "title_24dp_5F6368_FILL0_wght400_GRAD0_opsz24.svg",
    "classification-task": "category_24dp_5F6368_FILL0_wght300_GRAD0_opsz24.svg",
    "non-classification-task": "circle_24dp_5F6368_FILL0_wght300_GRAD0_opsz24.svg"
};

export function getIconUri(iconName: string): vscode.Uri {
    const relativeIconPath = iconPaths[iconName];
    if (relativeIconPath === undefined) {
        throw new Error(`Icon name ${iconName} not found`);
    }
    const resourceRootUri = vscode.Uri.file(config.GlobalStorageContext.resouceRoot);
    return vscode.Uri.joinPath(resourceRootUri, relativeIconPath);
}
