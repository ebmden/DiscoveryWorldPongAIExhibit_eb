
rm -r autojobs
mkdir autojobs
cd autojobs || exit

sbatch /home/bassoe/srdes/DiscoveryWorldPongAIExhibit/scripts/run_eb.sh

sleep 15

tail -f pong_job.out