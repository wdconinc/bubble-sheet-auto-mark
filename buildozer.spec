[app]
title = Bubble Sheet Auto-Mark
package.name = bubble_mark
package.domain = org.bubble_mark

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json
source.include_patterns = main.py,src/**

version = 0.1.0

requirements = python3==3.12,kivy==2.3.1,opencv,numpy,pillow

orientation = portrait,landscape

osx.python_version = 3

# Android specific
android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE
android.api = 33
android.minapi = 26
android.ndk = 25b
android.arch = arm64-v8a
android.build_tools_version = 34.0.0
android.accept_sdk_android_license = True

# iOS specific
ios.kivy_ios_url = https://github.com/kivy/kivy-ios
ios.kivy_ios_branch = master
ios.ios_deploy_url = https://github.com/phonegap/ios-deploy
ios.ios_deploy_branch = 1.10.0

[buildozer]
log_level = 2
warn_on_root = 1
