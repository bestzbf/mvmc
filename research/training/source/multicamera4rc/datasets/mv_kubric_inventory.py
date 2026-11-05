"""Validate the actual contents of the immutable official full-train release.

Archive names describe nominal ID ranges. Gaps may only be admitted with an
independent header inventory bound to all five publisher-checksummed archives.
The resulting exact ID set is still required of both the packs and GT audit.
"""
from __future__ import annotations

PUBLISHER_REVISION = 'cccb9128fb95d302c662151e65a09377175c2a3a'
PUBLISHED_ARCHIVES = (
    ('0031-1000', 394716348566, '515f369701a8c474880bc7b7fde1b8b4454e0198287b168848cd2a8aa0992ff8'),
    ('1001-2000', 404985432432, 'f1dfd394406f620cb567506de79bdad45456996000dd82a9ac322c829aca71dc'),
    ('2001-3000', 407247025781, '4117af637bab3b20f0656a00a6cd7ad2a6c56e689e3de70267acd6209d1c544c'),
    ('3001-4000', 407446344712, 'd6678c66bfffd2485f7bf01d0052dcfa2b3442c155d78eaa7d76076324e42793'),
    ('4001-5000', 406298411822, '967df23413878a06f5c5efdf169a5db711172687be588d03753df8c11c7cd28d'),
)


def expected_train_ids(manifest):
    evidence = manifest.get('raw_archive_inventory')
    if evidence is None:
        # Legacy manifests cannot waive even one ID using a count alone.
        return list(range(31, 5001))
    if (evidence.get('method') != 'independent_tar_headers_v1'
            or evidence.get('publisher_revision') != PUBLISHER_REVISION
            or not evidence.get('raw_and_packed_id_sets_equal')):
        raise ValueError('invalid independent raw archive inventory evidence')
    archives = evidence.get('archives', [])
    by_name = {a['name']: a for a in archives}
    if len(archives) != 5 or len(by_name) != 5:
        raise ValueError('raw archive inventory requires all five unique official archives')
    result = []
    for bounds, size, sha in PUBLISHED_ARCHIVES:
        name = f'kubric-multiview--train.full.{bounds}.tar.gz'
        row = by_name.get(name, {})
        lo, hi = map(int, bounds.split('-'))
        ids = row.get('scene_ids', [])
        if (row.get('bytes') != size or row.get('publisher_sha256') != sha
                or row.get('publisher_checksum_verified') is not True
                or row.get('header_scan_complete') is not True
                or not ids or any(type(i) is not int or not lo <= i <= hi for i in ids)
                or ids != sorted(set(ids))):
            raise ValueError(f'invalid publisher-bound raw inventory: {name}')
        result.extend(ids)
    return sorted(result)
