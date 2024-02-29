echo "Activating manual venv based on LCG_96python3"
source /cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh
SCRIPTDIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PIP_CONFIG_FILE="$SCRIPTDIR/pip.conf"
export PATH="$SCRIPTDIR/env/bin:${PATH}"
export PYTHONPATH="$SCRIPTDIR/env/lib/python3.6/site-packages:${PYTHONPATH}"
if [ -z "${_PS1_SET}" ]; then
    export PS1="(env) $PS1"
fi
export _PS1_SET=1
