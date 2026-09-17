# GAN-s
This was by far the most interesting project I've worked on in the nano degree We generated fake faces by making neural networks duel against each other. Getting the generator loss down to &lt;1 was my biggest hurdle. I tried training the generator twice and discriminator once and many other optimization techniques to get the loss to &lt;1 

cd /exp/FY26/AIxCC/ro31337/discver1/discver-patchengineer01
git init -q -b main
printf '__pycache__/\n*.pyc\n.pytest_cache/\n.crew/\ncrew.log\ndiscver/\noutput/\noss-crs/\ndiscver-patch-fix.zip\n' > .gitignore
git add -A && git commit -qm "baseline: code as of $(date +%F)"
git log --oneline -1



CREW_MAX_TURNS=120 nohup python microcrew.py --repo $(pwd) --parallel 3 --max-items 8 > crew.log 2>&1 &
tail -f crew.log
