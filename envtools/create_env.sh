
#!/bin/zsh

# run zsh script
# https://rowannicholls.github.io/bash/intro/passing_arguments.html
# zsh condition
# https://zsh.sourceforge.io/Doc/Release/Conditional-Expressions.html


#if [[ $1 =~ ^(-e|--env) ]]
#then 
#    envname="$2"
#fi


# help function
showhelp() {
    echo "
        please pass option:
        -v|--version 3.8 | 3.9 | 3.10 | 3.11
        -p|--path ~/VENV/test

        example: 
           source create_env.sh -p ~/VENV/test3.9 -v 3.9

        will create a python 3.9 venv at the path
        ~/VENV/test3.9   
    "
    # echo "
    #     please pass option:
    #     -e|--env "envname" to create a python env with this name
    #     -v|--version 3.8 | 3.9 | 3.10
    #     -p|--path 
    # "
}

show_venv_info() {
    echo "
    create a python venv
    version: $pythonversion
    at path: $envpath
    "
    #$envpath/$envname
}

show_info_activation() {
    echo "
    use 
        source $envpath/bin/activate
    to activate your env
    to deactivate with
        deactivate    
    "
}

# check whether user want to continue
check_user_agree() {
    show_venv_info
    read -q "usercontinue?Enter y|Y to continue... 
    "
    # save user input to variable $usercontinue
    # https://stackoverflow.com/questions/15174121/how-can-i-prompt-for-yes-no-style-confirmation-in-a-zsh-script/15174634#15174634
    if [[ $usercontinue =~ ^(y|Y) ]]; then
        programcontinue=true
    else
        programcontinue=false
    fi 
    echo "" # just echo a new line         
}

## generate cmd based on macos chip 
generate_cmd_macos() {
    # or x86_64
    if [[ $(uname -m) = "arm64" ]]; then
        myexec="/opt/homebrew/bin/python${pythonversion}"
    else
        # /usr/local/bin/python3.9, is a simlink
        myexec="/usr/local/bin/python${pythonversion}"
        # myexec="/usr/local/opt/python@${pythonversion}/bin/python3"
    fi
    mycmd="$myexec -m venv $envpath"
}

check_cmd_exists() {
    generate_cmd_macos
    # test only the location
    # https://stackoverflow.com/questions/7522712/how-can-i-check-if-a-command-exists-in-a-shell-script/7522866#7522866
    if [[ -f "$myexec" ]]; then
        hasexec=true
    else 
        hasexec=false
    fi
}

## create user env
create_env() {
    case $pythonversion in 
        3.8|3.9|3.10|3.11) 
            # echo "$mycmd"
            eval ${mycmd};
            # making env is successful
            if [[ $? -eq 0 ]]; then
                echo "activate the python venv $envpath..."
                eval "source $envpath/bin/activate";
                eval "python3 -m pip install --upgrade pip";
                eval "deactivate"
                echo "$envpath deactivated"
                show_info_activation
            fi
            ;;
    esac
}

## execute the main program
exec_main() {
    # create mycmd and myexec
    check_cmd_exists
    if [[ "$programcontinue" = true ]]; then
        if [[ "$hasexec" = true ]]; then
            create_env
        else
            echo "
            $myexec
            doesn't exists on your mac.
            " 
        fi 
    else 
        echo "program end by user."    
    fi
}

## check the user inputs
while [[ "$#" -gt 0 ]]
do case $1 in 
    #-e|--env) envname="$2"
    #shift;;
    -v|--version) pythonversion="$2"
    shift;;
    -p|--path) envpath="$2"
    shift;;
    *) showhelp
    # exit 0;; 
    # not using exit 1, the vscode termin will crash
    # also not exit 0;; the terminal will close
esac
shift
done
# https://rowannicholls.github.io/bash/intro/passing_arguments.html

# if one of the variable has zero length
# if [[ -z "$envname" || -z "$pythonversion" || -z "$envpath" ]]; then 
if [[ -z "$pythonversion" || -z "$envpath" ]]; then
    showhelp
else
    if ! [[ $pythonversion =~ ^(3.8|3.9|3.10|3.11) ]]; then
        echo "-v|--version can only be one of 3.8|3.9|3.10|3.11"
    else
        check_user_agree
        exec_main
    fi    
fi
