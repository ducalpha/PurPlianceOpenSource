from pathlib import Path
import json
import os
import sys

from oppnlp.analyze.priv_stmt.priv_stmt_extractor_impl import PrivStmtExtractorImpl


def extract_and_write_priv_stmts(policy_file: Path, out_dir: Path):
    """Extract privacy statements from a single file and write them to disk."""
    assert policy_file.exists(), f'{policy_file} not exist.'
    print(f'Processing {policy_file}')

    pkg_name = policy_file.stem
    priv_stmts_file = out_dir / f'{pkg_name}.json'

    priv_stmts = PrivStmtExtractorImpl().extract_priv_stmts_from_file(policy_file, single_sent_txt=True)

    # Convert everything to string for serialization.
    for p in priv_stmts:
        for k, v in p.items():
            p[k] = str(v)

    # priv_stmts is a list of dicts.
    with priv_stmts_file.open('w') as f:
        json.dump(priv_stmts, f, indent=4)


def get_priv_stmts(policy_file_or_files, out_dir: Path):
    """Extract privacy statements from a set of files."""
    for policy_file in sorted(list(policy_file_or_files)):
        extract_and_write_priv_stmts(policy_file, out_dir)


def main():
    """Get privacy statements in a doc_dir."""
    if len(sys.argv) < 1:
        print(f'Usage: {sys.argv[0]} <doc_dir>')
        print(f'doc_dir: contains plain sentencized text files *.txt to be analyzed.')
        print(f'Each non-blank line in the file should contain one and only 1 sentence.')

    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        doc_dir = Path(sys.argv[1])
        assert doc_dir.exists(), f'{doc_dir} not exist'
    else:
        doc_dir = Path(os.getcwd())

    out_dir = doc_dir / 'stmt'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output dir: {out_dir}')

    assert doc_dir.exists(), f'{doc_dir} does not exist.'

    policy_files = list(doc_dir.glob('*.txt'))

    num_policy_files = len(policy_files)
    if num_policy_files == 0:
        print('No file to process.')
        return

    print(f'Num policy files: {num_policy_files}')
    get_priv_stmts(policy_files, out_dir)


if __name__ == '__main__':
    main()
