import { existsSync, statSync } from 'fs';

export function isDirectory(path: string): boolean {
    return existsSync(path) && statSync(path).isDirectory();
}
