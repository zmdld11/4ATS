from pathlib import Path
import librosa

# 获取当前文件所在目录
current_file = Path(__file__)
# 获取music文件夹路径
music_dir = current_file.parent.parent / 'music'

def get_bpm_from_music_folder(filename):
    audio_path = music_dir / filename
    
    if not audio_path.exists():
        print(f"文件不存在: {audio_path}")
        return None
    
    y, sr = librosa.load(str(audio_path))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    print(tempo)
    return tempo

# 使用示例
bpm = get_bpm_from_music_folder("2.flac")
# bpm = get_bpm_from_music_folder("1.mp3")