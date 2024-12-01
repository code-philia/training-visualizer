import * as vscode from 'vscode';
import * as constants from './constants';
import { setAsDataFolderAndLoadVisualizationResult, startVisualization } from './processes/startVisualization';
import { setDataFolder } from './processes/configVisualization';

// TODO this should only be called once
// but now it has not limitation
export function doCommandsRegistration(): vscode.Disposable {
    const commandsRegistration = vscode.Disposable.from(
        vscode.commands.registerCommand(constants.CommandID.setAsDataFolderAndLoadVisualizationResult, setAsDataFolderAndLoadVisualizationResult),
        vscode.commands.registerCommand(constants.CommandID.setAsDataFolder, (file: any) => {
            if (!(file instanceof vscode.Uri)) {
                return;
            }
            setDataFolder(file);
        }),
        vscode.commands.registerCommand(constants.CommandID.loadVisualization, startVisualization),
        vscode.commands.registerCommand(constants.CommandID.configureAndLoadVisualization, () => { startVisualization(true); })
    );
    return commandsRegistration;
}
