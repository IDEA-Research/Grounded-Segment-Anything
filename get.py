def git_clone(repo='positive666/Tag2Text', branch='main'):
    from pathlib import Path
    import os 
    if not os.path.exists('Tag2Text'): #如果目录不存在就返回False
        
        
        from subprocess import check_output
        from pathlib import Path
        url = f'https://github.com/{repo}'
        check_output(f'git clone {url}.git -b {branch}', shell=True)  # git fetch
from pathlib import Path
 
git_clone()