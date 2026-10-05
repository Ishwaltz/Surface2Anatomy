import argparse, ast, csv, hashlib, json, os, pickletools, zipfile
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
root = a.root.resolve()
out = root / 'reports/ipmi121'
audit = out / 'audit'
audit.mkdir(parents=True, exist_ok=True)

def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)

def read_csv(rel):
    with (root / rel).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

inventory, hits, binaries, errors = [], [], [], []
terms = ('uterus', 'ovary', 'ovaries', 'vagina')
for base, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '.next', '__pycache__', '.venv', 'venv'}]
    for name in files:
        path = Path(base) / name
        rel = path.relative_to(root).as_posix()
        if rel.startswith('reports/ipmi121/'):
            continue
        ext = path.suffix.lower()
        if ext not in {'.csv','.json','.pt','.pth','.pkl','.pickle','.py','.md','.txt'} and not any(t in name.lower() for t in ('manifest','ontology','segmentation','label_map','target_map')):
            continue
        inventory.append({'path':rel, 'bytes':path.stat().st_size})
        try:
            if ext in {'.pt','.pth','.pkl','.pickle'}:
                strings = set()
                if zipfile.is_zipfile(path):
                    with zipfile.ZipFile(path) as z:
                        parts = [n for n in z.namelist() if n.endswith('data.pkl')]
                        blobs = [z.read(n) for n in parts]
                elif ext in {'.pkl','.pickle'}:
                    blobs = [path.read_bytes()]
                else:
                    blobs = []
                for blob in blobs:
                    for op, arg, pos in pickletools.genops(blob):
                        if isinstance(arg, str):
                            strings.add(arg)
                found = sorted(s for s in strings if any(t in s.lower() for t in terms))
                binaries.append({'path':rel,'inspection':'STATIC_PICKLE_STRINGS_ONLY' if blobs else 'UNSUPPORTED_SERIALIZATION', 'female_strings':json.dumps(found)})
            elif ext in {'.csv','.json','.py','.md','.txt'}:
                for i, line in enumerate(path.read_text(encoding='utf-8-sig', errors='replace').splitlines(), 1):
                    if any(t in line.lower() for t in terms):
                        hits.append({'path':rel,'line':i,'text':line[:1500]})
        except Exception as e:
            errors.append({'path':rel,'error':str(e)})

write_csv(audit/'source_inventory.csv', inventory, ['path','bytes'])
write_csv(audit/'female_label_source_hits.csv', hits, ['path','line','text'])
write_csv(audit/'serialized_metadata_inventory.csv', binaries, ['path','inspection','female_strings'])
write_csv(audit/'inspection_errors.csv', errors, ['path','error'])
maps = []
for rel in ('labels.py','sharon/labels.py','hf_space/sharon/labels.py'):
    tree = ast.parse((root/rel).read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TOTAL_CLASSES_121' for t in node.targets):
            mapping = ast.literal_eval(node.value)
            maps.append((rel, mapping))
mapping = maps[0][1]
assert len(mapping)==121 and set(mapping.values())==set(range(1,122))
assert maps[1][1] == mapping
defs = {r['target_name']:r for r in read_csv('reports/phase1/02_target_definition_table.csv')}
provs = {r['target_name']:r for r in read_csv('reports/phase1r/target_provenance_v2.csv')}
rows = []
for name, label in mapping.items():
    synthetic = provs[name]['provenance']=='SYNTHETIC'
    rows.append(dict(target_index=label-1, canonical_name=name,
        anatomical_group=defs.get(name,{}).get('anatomical_group','UNVERIFIED'),
        laterality='left' if '_left' in name else 'right' if '_right' in name else 'none',
        sex_specific='female' if label>=118 else 'male' if name=='prostate' else 'no',
        label_source='reports/phase1r/target_provenance_v2.csv',
        train_support='NOT_ESTABLISHED',val_support='NOT_ESTABLISHED',test_support='NOT_ESTABLISHED',external_support='NOT_VERIFIED',
        status='BLOCKED_SYNTHETIC_LEGACY_LABEL' if synthetic else 'CANDIDATE_PENDING_NEW_AUDIT',
        notes='Candidate ontology, not a validated benchmark. '+provs[name]['notes']))
write_csv(out/'01_TARGET_ONTOLOGY_121.csv',rows,list(rows[0]))
sources = ['labels.py','sharon/labels.py','dataset_preprocessing.py','sharon/dataset_preprocessing.py','tools/reconstruct/target_extractor.py','reports/phase1r/target_provenance_v2.csv','reports/phase1r/target_support_matrix.csv','sharon/dataset_v3/target_ontology_v3.csv','data_external/dap_atlas/raw/label_name.csv','reports/phase11/mappings/AMOS_target_mapping.csv']
write_csv(audit/'evidence_sha256.csv',[{'path':r,'sha256':hashlib.sha256((root/r).read_bytes()).hexdigest()} for r in sources],['path','sha256'])
summary = dict(study_version='Surface2Anatomy-IPMI-121-v1', status='STOPPED_AT_ONTOLOGY_GATE', declared_target_count=121, target_count_verified=False, blocked_legacy_targets=[r['canonical_name'] for r in rows if r['status'].startswith('BLOCKED')], final_headline=None, data_audit='NOT_PASSED', final_test_locked=False, final_test_evaluated=False, inventory_files=len(inventory), serialized_files=len(binaries), inspection_errors=len(errors))
(audit/'ontology_gate.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
