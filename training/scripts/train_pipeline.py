from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def resolve_model(model_value: str, config_path: Path) -> str:
    model_path = Path(model_value)
    if model_path.is_absolute():
        return str(model_path)
    if model_value.endswith('.pt') and not model_value.startswith('../'):
        return model_value
    return str((config_path.parent / model_value).resolve())


def run_stage(config_path: Path):
    with config_path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model_value = cfg.pop('model')
    model = YOLO(resolve_model(model_value, config_path))
    data_value = cfg.get('data')
    if data_value:
        cfg['data'] = str((config_path.parent / data_value).resolve())
    project_value = cfg.get('project')
    if project_value:
        cfg['project'] = str((config_path.parent / project_value).resolve())
    print(f'\n>>> 开始训练: {config_path.name}')
    print(cfg)
    model.train(**cfg)


def load_profile(profile_name: str) -> tuple[Path, Path]:
    profile_path = ROOT / 'configs' / f'{profile_name}.yaml'
    if not profile_path.exists():
        raise FileNotFoundError(f'未找到训练配置档案: {profile_path}')
    profile = yaml.safe_load(profile_path.read_text(encoding='utf-8'))
    return ROOT / 'configs' / profile['stage1'], ROOT / 'configs' / profile['stage2']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='两阶段训练入口')
    parser.add_argument('--profile', default='gtx1650_n', choices=['gtx1650_n', 'gtx1650_s'], help='训练配置档案')
    args = parser.parse_args()
    stage1, stage2 = load_profile(args.profile)
    print(f'使用训练档案: {args.profile}')
    run_stage(stage1)
    run_stage(stage2)
