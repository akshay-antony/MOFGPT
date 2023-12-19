{
echo "# Force color prompt"
echo "force_color_prompt=yes"

echo "# Check for interactive shell"
echo "case \$- in"
echo "    *i*) ;;"
echo "      *) return;;"
echo "esac"

echo "# Enable color support for ls"
echo "if [ -x /usr/bin/dircolors ]; then"
echo "    test -r ~/.dircolors && eval \"\$(dircolors -b ~/.dircolors)\" || eval \"\$(dircolors -b)\""
echo "    alias ls='ls --color=auto'"
echo "fi"

echo "# Set a fancy prompt (non-color, unless we know we \"want\" color)"
echo "case \"\$TERM\" in"
echo "    xterm-color|*-256color) color_prompt=yes;;"
echo "esac"

echo "# Set the colored PS1"
echo "if [ \"\$color_prompt\" = yes ]; then"
echo "    PS1='\${debian_chroot:+(\$debian_chroot)}\\[\\033[01;36m\\]\\u@\\h\\[\\033[00m\\]:\\[\\033[01;35m\\]\\w\\[\\033[00m\\]\\\$ '"
echo "else"
echo "    PS1='\${debian_chroot:+(\$debian_chroot)}\\u@\\h:\\w\\\$ '"
echo "fi"
} >> /root/.bashrc
