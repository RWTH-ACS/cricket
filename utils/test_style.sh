#! /bin/bash

echo "checking all files for style errors"

TMPFILE=`mktemp`

check_format () {
  echo -n "Checking $1: "
  diff -u $1 <(clang-format  $1) > $TMPFILE
  if [ $? -ne 0 ]; then
    echo "ERROR $1 does not adhere the style guide, diff to well formated style:"
    cat $TMPFILE
    echo ""
    STYLE_ERRORS=1
  else
    echo "Ok"
  fi

}

for f in src/*.c ; do
  check_format $f
done

for f in include/*.h ; do
  check_format $f
done

if [ -n "$STYLE_ERRORS" ]; then
  echo "There were style errors"
  exit -1
fi
echo "Success: All source files have passed the style check"
rm $TMPFILE
