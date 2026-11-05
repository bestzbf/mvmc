"""Controller-only JSON-frame fix; native update acceptance is unchanged."""
import json


def top_level_audit(native_base):
    class TopLevelUpdateAudit(native_base):
        def poll(self):
            if not self.path.exists():return
            stat=self.path.stat();identity=(stat.st_dev,stat.st_ino)
            if (self.identity is not None and identity!=self.identity) or stat.st_size<self.offset:
                raise ValueError('live training log was replaced or truncated')
            self.identity=identity
            with self.path.open('rb') as stream:
                stream.seek(self.offset);raw=self.pending+stream.read();self.offset=stream.tell()
            complete,separator,self.pending=raw.rpartition(b'\n')
            if not separator:self.pending=raw;return
            decoder=json.JSONDecoder()
            for line in complete.decode('utf-8').splitlines():
                position=0
                while True:
                    begin=line.find('{',position)
                    if begin<0:break
                    try:row,position=decoder.raw_decode(line,begin)
                    except json.JSONDecodeError:break
                    # Consume the whole outer frame, never regex-search nested
                    # data_migration/checkpoint metadata for another step.
                    if not isinstance(row,dict) or 'step' not in row:continue
                    if not any(key in row for key in ('world_size','rank_batches','gradient_accumulation','optimizer_step_skipped')):continue
                    # Genuine update envelopes retain the original mandatory
                    # fields, strict+1 ordering, finite checks and6-rank proof.
                    self.accept(row)
    return TopLevelUpdateAudit
