"""Read local W&B history and estimate BC progress; does not touch training."""
import argparse
import datetime
import json
from pathlib import Path

import yaml
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', nargs='?')
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
runs = root / 'RL-100/data/outputs'
run = Path(args.run_dir) if args.run_dir else max(
    runs.glob('adroit_door_bc_*'), key=lambda p: p.stat().st_mtime)
cfg = yaml.safe_load((run / '.hydra/config.yaml').read_text())
history_files = list(run.glob('wandb/offline-run-*/run-*.wandb'))
source = max(history_files, key=lambda p: p.stat().st_mtime)
store = DataStore()
# The installed W&B scanner opens files with r+b even though scanning only reads.
store._fname = str(source)
store._fp = source.open('rb')
store._index = 0
store._size_bytes = source.stat().st_size
store._opened_for_scan = True
store._read_header()
last = None
evaluations = []
exit_code = None
try:
    while True:
        block = store.scan_data()
        if block is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(block)
        if record.HasField('exit'):
            exit_code = record.exit.exit_code
        if record.HasField('history'):
            row = {item.key or '.'.join(item.nested_key): json.loads(item.value_json)
                   for item in record.history.item}
            if 'epoch' in row:
                last = row
            if 'test_mean_score' in row:
                evaluations.append(row)
except Exception as exc:
    print(f'最后一条实时日志尚未写完整，使用此前记录：{type(exc).__name__}')
finally:
    store._fp.close()
if last is None:
    raise SystemExit('尚无训练记录。')
total_epochs = cfg['training']['num_epochs']
epoch = int(last['epoch'])
display_tz = datetime.timezone(datetime.timedelta(hours=8))
now = datetime.datetime.now(display_tz)
print(f'检查时间：{now:%Y-%m-%d %H:%M:%S} 北京时间')
print(f'输出目录：{run}')
print(f'当前 epoch：{epoch + 1}/{total_epochs}（代码编号 {epoch}）')
print(f'已记录 global_step：{last.get("global_step", "?")}')
print(f'当前训练 loss：{last.get("train_loss", float("nan")):.5f}')
if '_runtime' in last:
    print(f'日志记录已运行：{last["_runtime"] / 60:.1f} 分钟')
if exit_code is not None:
    print(f'训练进程已退出，W&B 退出码：{exit_code}')
if evaluations:
    latest = evaluations[-1]
    best = max(evaluations, key=lambda x: x['test_mean_score'])
    print(f'最新评估：epoch {latest["epoch"] + 1}，成功率 {latest["test_mean_score"]:.1%}')
    print(f'最佳评估：epoch {best["epoch"] + 1}，成功率 {best["test_mean_score"]:.1%}')
rates = [(b['_runtime'] - a['_runtime']) / (b['epoch'] - a['epoch'])
         for a, b in zip(evaluations, evaluations[1:]) if b['epoch'] > a['epoch']]
if rates and exit_code is None:
    seconds_per_epoch = sum(rates[-3:]) / len(rates[-3:])
    remaining = max(0, total_epochs - epoch - 1) * seconds_per_epoch
    finish = now + datetime.timedelta(seconds=remaining)
    print(f'预计剩余：约 {remaining / 60:.1f} 分钟（含周期评估，可能波动）')
    print(f'预计结束：{finish:%Y-%m-%d %H:%M:%S} 北京时间')
updated = datetime.datetime.fromtimestamp(source.stat().st_mtime, display_tz)
print(f'日志更新时间：{updated:%Y-%m-%d %H:%M:%S} 北京时间')
if exit_code is None and (now - updated).total_seconds() > 120:
    print('日志超过两分钟未更新；可能在评估、已结束或异常，请同时检查训练终端。')
