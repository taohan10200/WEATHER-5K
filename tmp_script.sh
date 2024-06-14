sleep 10s
echo "This command is excuted on comput node!!!"
code-server serve-local --host 0.0.0.0 --port 10086 &
ngrok http --domain=crowd.ngrok.app http://localhost:10086
# kill all processes with run.py
ps aux | grep run.py | grep -v grep | awk '{print $2}' | xargs kill
pgrep -f -- --gpu\ 4 | xargs -I{} kill {}
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
