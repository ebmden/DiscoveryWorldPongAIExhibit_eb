rm -r autojobs
mkdir autojobs
cd autojobs || exit

sbatch /home/bassoe/srdes/DiscoveryWorldPongAIExhibit/run_eb.sh
#sbatch --time=1440 /home/bassoe/srdes/DiscoveryWorldPongAIExhibit/train_job.bat

sleep 15

tail -f pong_job.out