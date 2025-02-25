; ModuleID = '/app/example.c'
source_filename = "/app/example.c"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-unknown-elf"

;define void @memrefCopy(i32 %0, ptr %1, ptr %2) {
;  ret void
;}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none)
define dso_local void @memrefCopy32bit(ptr nocapture noundef readonly %0, ptr nocapture noundef readonly %1) local_unnamed_addr #0 !dbg !12 {
;    #dbg_value(ptr %0, !38, !DIExpression(), !48)
;    #dbg_value(ptr %1, !39, !DIExpression(), !48)
;    #dbg_value(i32 0, !40, !DIExpression(), !49)
  %3 = getelementptr inbounds i8, ptr %0, i32 12
;    #dbg_value(i32 0, !40, !DIExpression(), !49)
  %4 = load i32, ptr %3, align 4, !dbg !50, !tbaa !51
  %5 = icmp eq i32 %4, 0, !dbg !55
  br i1 %5, label %26, label %6, !dbg !56

6:                                                ; preds = %2
  %7 = getelementptr inbounds i8, ptr %0, i32 16
  %8 = getelementptr inbounds i8, ptr %0, i32 4
  %9 = getelementptr inbounds i8, ptr %0, i32 8
  %10 = getelementptr inbounds i8, ptr %0, i32 20
  %11 = getelementptr inbounds i8, ptr %0, i32 24
  %12 = getelementptr inbounds i8, ptr %1, i32 4
  %13 = getelementptr inbounds i8, ptr %1, i32 8
  %14 = getelementptr inbounds i8, ptr %1, i32 20
  %15 = getelementptr inbounds i8, ptr %1, i32 24
  %16 = load i32, ptr %7, align 4, !dbg !57, !tbaa !51
  %17 = icmp eq i32 %16, 0, !dbg !59
  br i1 %17, label %26, label %18

18:                                               ; preds = %6, %29
  %19 = phi i32 [ %30, %29 ], [ %4, %6 ]
  %20 = phi i32 [ %31, %29 ], [ 1, %6 ], !dbg !57
  %21 = phi i32 [ %32, %29 ], [ 0, %6 ]
 ;   #dbg_value(i32 %21, !40, !DIExpression(), !49)
 ;   #dbg_value(i32 0, !44, !DIExpression(), !60)
  %22 = icmp eq i32 %20, 0, !dbg !59
  br i1 %22, label %29, label %23, !dbg !61

23:                                               ; preds = %18
  %24 = load ptr, ptr %8, align 4, !tbaa !62
  %25 = load ptr, ptr %12, align 4, !tbaa !62
  br label %34, !dbg !61

26:                                               ; preds = %29, %6, %2
  ret void, !dbg !65

27:                                               ; preds = %34
  %28 = load i32, ptr %3, align 4, !dbg !50, !tbaa !51
  br label %29, !dbg !66

29:                                               ; preds = %27, %18
  %30 = phi i32 [ %28, %27 ], [ %19, %18 ], !dbg !50
  %31 = phi i32 [ %54, %27 ], [ 0, %18 ]
  %32 = add nuw i32 %21, 1, !dbg !66
 ;   #dbg_value(i32 %32, !40, !DIExpression(), !49)
  %33 = icmp ult i32 %32, %30, !dbg !55
  br i1 %33, label %18, label %26, !dbg !56, !llvm.loop !67

34:                                               ; preds = %23, %34
  %35 = phi i32 [ 0, %23 ], [ %53, %34 ]
 ;   #dbg_value(i32 %35, !44, !DIExpression(), !60)
  %36 = load i32, ptr %9, align 4, !dbg !71, !tbaa !73
  %37 = load i32, ptr %10, align 4, !dbg !74, !tbaa !51
  %38 = mul i32 %37, %21, !dbg !75
  %39 = load i32, ptr %11, align 4, !dbg !76, !tbaa !51
  %40 = mul i32 %39, %35, !dbg !77
  %41 = getelementptr i32, ptr %24, i32 %36, !dbg !78
  %42 = getelementptr i32, ptr %41, i32 %38, !dbg !78
  %43 = getelementptr i32, ptr %42, i32 %40, !dbg !78
  %44 = load i32, ptr %43, align 4, !dbg !78, !tbaa !51
  %45 = load i32, ptr %13, align 4, !dbg !79, !tbaa !73
  %46 = load i32, ptr %14, align 4, !dbg !80, !tbaa !51
  %47 = mul i32 %46, %21, !dbg !81
  %48 = load i32, ptr %15, align 4, !dbg !82, !tbaa !51
  %49 = mul i32 %48, %35, !dbg !83
  %50 = getelementptr i32, ptr %25, i32 %45, !dbg !84
  %51 = getelementptr i32, ptr %50, i32 %47, !dbg !84
  %52 = getelementptr i32, ptr %51, i32 %49, !dbg !84
  store i32 %44, ptr %52, align 4, !dbg !85, !tbaa !51
  %53 = add nuw i32 %35, 1, !dbg !86
  ;  #dbg_value(i32 %53, !44, !DIExpression(), !60)
  %54 = load i32, ptr %7, align 4, !dbg !57, !tbaa !51
  %55 = icmp ult i32 %53, %54, !dbg !59
  br i1 %55, label %34, label %27, !dbg !61, !llvm.loop !87
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none)
define dso_local void @memrefCopy8bit(ptr nocapture noundef readonly %0, ptr nocapture noundef readonly %1) local_unnamed_addr #0 !dbg !89 {
 ;   #dbg_value(ptr %0, !106, !DIExpression(), !114)
 ;   #dbg_value(ptr %1, !107, !DIExpression(), !114)
 ;   #dbg_value(i32 0, !108, !DIExpression(), !115)
  %3 = getelementptr inbounds i8, ptr %0, i32 12
 ;   #dbg_value(i32 0, !108, !DIExpression(), !115)
  %4 = load i32, ptr %3, align 4, !dbg !116, !tbaa !51
  %5 = icmp eq i32 %4, 0, !dbg !117
  br i1 %5, label %23, label %6, !dbg !118

6:                                                ; preds = %2
  %7 = getelementptr inbounds i8, ptr %0, i32 16
  %8 = getelementptr inbounds i8, ptr %0, i32 4
  %9 = getelementptr inbounds i8, ptr %0, i32 8
  %10 = getelementptr inbounds i8, ptr %0, i32 20
  %11 = getelementptr inbounds i8, ptr %0, i32 24
  %12 = getelementptr inbounds i8, ptr %1, i32 4
  %13 = getelementptr inbounds i8, ptr %1, i32 8
  %14 = getelementptr inbounds i8, ptr %1, i32 20
  %15 = getelementptr inbounds i8, ptr %1, i32 24
  %16 = load i32, ptr %7, align 4, !dbg !119, !tbaa !51
  %17 = icmp eq i32 %16, 0, !dbg !121
  br i1 %17, label %23, label %18

18:                                               ; preds = %6, %26
  %19 = phi i32 [ %27, %26 ], [ %4, %6 ]
  %20 = phi i32 [ %28, %26 ], [ 1, %6 ], !dbg !119
  %21 = phi i32 [ %29, %26 ], [ 0, %6 ]
 ;   #dbg_value(i32 %21, !108, !DIExpression(), !115)
 ;   #dbg_value(i32 0, !110, !DIExpression(), !122)
  %22 = icmp eq i32 %20, 0, !dbg !121
  br i1 %22, label %26, label %31, !dbg !123

23:                                               ; preds = %26, %6, %2
  ret void, !dbg !124

24:                                               ; preds = %31
  %25 = load i32, ptr %3, align 4, !dbg !116, !tbaa !51
  br label %26, !dbg !125

26:                                               ; preds = %24, %18
  %27 = phi i32 [ %25, %24 ], [ %19, %18 ], !dbg !116
  %28 = phi i32 [ %53, %24 ], [ 0, %18 ]
  %29 = add nuw i32 %21, 1, !dbg !125
 ;   #dbg_value(i32 %29, !108, !DIExpression(), !115)
  %30 = icmp ult i32 %29, %27, !dbg !117
  br i1 %30, label %18, label %23, !dbg !118, !llvm.loop !126

31:                                               ; preds = %18, %31
  %32 = phi i32 [ %52, %31 ], [ 0, %18 ]
 ;   #dbg_value(i32 %32, !110, !DIExpression(), !122)
  %33 = load ptr, ptr %8, align 4, !dbg !128, !tbaa !130
  %34 = load i32, ptr %9, align 4, !dbg !132, !tbaa !133
  %35 = load i32, ptr %10, align 4, !dbg !134, !tbaa !51
  %36 = mul i32 %35, %21, !dbg !135
  %37 = load i32, ptr %11, align 4, !dbg !136, !tbaa !51
  %38 = mul i32 %37, %32, !dbg !137
  %39 = getelementptr i8, ptr %33, i32 %34, !dbg !138
  %40 = getelementptr i8, ptr %39, i32 %36, !dbg !138
  %41 = getelementptr i8, ptr %40, i32 %38, !dbg !138
  %42 = load i8, ptr %41, align 1, !dbg !138, !tbaa !139
  %43 = load ptr, ptr %12, align 4, !dbg !140, !tbaa !130
  %44 = load i32, ptr %13, align 4, !dbg !141, !tbaa !133
  %45 = load i32, ptr %14, align 4, !dbg !142, !tbaa !51
  %46 = mul i32 %45, %21, !dbg !143
  %47 = load i32, ptr %15, align 4, !dbg !144, !tbaa !51
  %48 = mul i32 %47, %32, !dbg !145
  %49 = getelementptr i8, ptr %43, i32 %44, !dbg !146
  %50 = getelementptr i8, ptr %49, i32 %46, !dbg !146
  %51 = getelementptr i8, ptr %50, i32 %48, !dbg !146
  store i8 %42, ptr %51, align 1, !dbg !147, !tbaa !139
  %52 = add nuw i32 %32, 1, !dbg !148
 ;   #dbg_value(i32 %52, !110, !DIExpression(), !122)
  %53 = load i32, ptr %7, align 4, !dbg !119, !tbaa !51
  %54 = icmp ult i32 %52, %53, !dbg !121
  br i1 %54, label %31, label %24, !dbg !123, !llvm.loop !149
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) "approx-func-fp-math"="true" "no-builtin-printf" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic-rv32" "target-features"="+32bit,+a,+d,+f,+m,+relax,+zfh,+zfhmin,+zicsr,+zmmul,-b,-c,-e,-experimental-smmpm,-experimental-smnpm,-experimental-ssnpm,-experimental-sspm,-experimental-ssqosid,-experimental-supm,-experimental-zacas,-experimental-zalasr,-experimental-zicfilp,-experimental-zicfiss,-h,-shcounterenw,-shgatpa,-shtvala,-shvsatpa,-shvstvala,-shvstvecd,-smaia,-smcdeleg,-smcsrind,-smepmp,-smstateen,-ssaia,-ssccfg,-ssccptr,-sscofpmf,-sscounterenw,-sscsrind,-ssstateen,-ssstrict,-sstc,-sstvala,-sstvecd,-ssu64xl,-svade,-svadu,-svbare,-svinval,-svnapot,-svpbmt,-v,-xcvalu,-xcvbi,-xcvbitmanip,-xcvelw,-xcvmac,-xcvmem,-xcvsimd,-xsfcease,-xsfvcp,-xsfvfnrclipxfqf,-xsfvfwmaccqqq,-xsfvqmaccdod,-xsfvqmaccqoq,-xsifivecdiscarddlone,-xsifivecflushdlone,-xtheadba,-xtheadbb,-xtheadbs,-xtheadcmo,-xtheadcondmov,-xtheadfmemidx,-xtheadmac,-xtheadmemidx,-xtheadmempair,-xtheadsync,-xtheadvdot,-xventanacondops,-xwchc,-za128rs,-za64rs,-zaamo,-zabha,-zalrsc,-zama16b,-zawrs,-zba,-zbb,-zbc,-zbkb,-zbkc,-zbkx,-zbs,-zca,-zcb,-zcd,-zce,-zcf,-zcmop,-zcmp,-zcmt,-zdinx,-zfa,-zfbfmin,-zfinx,-zhinx,-zhinxmin,-zic64b,-zicbom,-zicbop,-zicboz,-ziccamoa,-ziccif,-zicclsm,-ziccrse,-zicntr,-zicond,-zifencei,-zihintntl,-zihintpause,-zihpm,-zimop,-zk,-zkn,-zknd,-zkne,-zknh,-zkr,-zks,-zksed,-zksh,-zkt,-ztso,-zvbb,-zvbc,-zve32f,-zve32x,-zve64d,-zve64f,-zve64x,-zvfbfmin,-zvfbfwma,-zvfh,-zvfhmin,-zvkb,-zvkg,-zvkn,-zvknc,-zvkned,-zvkng,-zvknha,-zvknhb,-zvks,-zvksc,-zvksed,-zvksg,-zvksh,-zvkt,-zvl1024b,-zvl128b,-zvl16384b,-zvl2048b,-zvl256b,-zvl32768b,-zvl32b,-zvl4096b,-zvl512b,-zvl64b,-zvl65536b,-zvl8192b" "unsafe-fp-math"="true" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !8, !9, !10}
!llvm.ident = !{!11}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 20.0.0git (https://github.com/llvm/llvm-project.git 8644a2aa0f3540c69464f56b3d538e888b6cbdcb)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/app/example.c", directory: "/app", checksumkind: CSK_MD5, checksum: "e539ef01c7473da6adac29899f664f36")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 1, !"target-abi", !"ilp32d"}
!6 = !{i32 6, !"riscv-isa", !7}
!7 = !{!"rv32i2p1_m2p0_a2p1_f2p2_d2p2_zicsr2p0_zmmul1p0_zfh1p0_zfhmin1p0"}
!8 = !{i32 1, !"Code Model", i32 3}
!9 = !{i32 8, !"SmallDataLimit", i32 8}
!10 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!11 = !{!"clang version 20.0.0git (https://github.com/llvm/llvm-project.git 8644a2aa0f3540c69464f56b3d538e888b6cbdcb)"}
!12 = distinct !DISubprogram(name: "memrefCopy32bit", scope: !13, file: !13, line: 26, type: !14, scopeLine: 26, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !37)
!13 = !DIFile(filename: "example.c", directory: "/app", checksumkind: CSK_MD5, checksum: "e539ef01c7473da6adac29899f664f36")
!14 = !DISubroutineType(types: !15)
!15 = !{null, !16, !16}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !17, size: 32)
!17 = !DIDerivedType(tag: DW_TAG_typedef, name: "TwoDMemrefI32_t", file: !13, line: 24, baseType: !18)
!18 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TwoDMemrefI32", file: !13, line: 4, size: 224, elements: !19)
!19 = !{!20, !27, !28, !32, !36}
!20 = !DIDerivedType(tag: DW_TAG_member, name: "data", scope: !18, file: !13, line: 5, baseType: !21, size: 32)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 32)
!22 = !DIDerivedType(tag: DW_TAG_typedef, name: "int32_t", file: !23, line: 44, baseType: !24)
!23 = !DIFile(filename: "/opt/compiler-explorer/riscv32/gcc-10.2.0/riscv32-unknown-elf/lib/gcc/riscv32-unknown-elf/10.2.0/../../../../riscv32-unknown-elf/include/sys/_stdint.h", directory: "", checksumkind: CSK_MD5, checksum: "ab914e287601b2385e57880e6599aa6b")
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int32_t", file: !25, line: 77, baseType: !26)
!25 = !DIFile(filename: "/opt/compiler-explorer/riscv32/gcc-10.2.0/riscv32-unknown-elf/lib/gcc/riscv32-unknown-elf/10.2.0/../../../../riscv32-unknown-elf/include/machine/_default_types.h", directory: "", checksumkind: CSK_MD5, checksum: "f7024d0682a918b41f94e8be9cd90461")
!26 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!27 = !DIDerivedType(tag: DW_TAG_member, name: "aligned_data", scope: !18, file: !13, line: 7, baseType: !21, size: 32, offset: 32)
!28 = !DIDerivedType(tag: DW_TAG_member, name: "offset", scope: !18, file: !13, line: 9, baseType: !29, size: 32, offset: 64)
!29 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", file: !23, line: 48, baseType: !30)
!30 = !DIDerivedType(tag: DW_TAG_typedef, name: "__uint32_t", file: !25, line: 79, baseType: !31)
!31 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!32 = !DIDerivedType(tag: DW_TAG_member, name: "shape", scope: !18, file: !13, line: 10, baseType: !33, size: 64, offset: 96)
!33 = !DICompositeType(tag: DW_TAG_array_type, baseType: !29, size: 64, elements: !34)
!34 = !{!35}
!35 = !DISubrange(count: 2)
!36 = !DIDerivedType(tag: DW_TAG_member, name: "stride", scope: !18, file: !13, line: 11, baseType: !33, size: 64, offset: 160)
!37 = !{!38, !39, !40, !44}
!38 = !DILocalVariable(name: "src", arg: 1, scope: !12, file: !13, line: 26, type: !16)
!39 = !DILocalVariable(name: "dst", arg: 2, scope: !12, file: !13, line: 26, type: !16)
!40 = !DILocalVariable(name: "row", scope: !41, file: !13, line: 27, type: !42)
!41 = distinct !DILexicalBlock(scope: !12, file: !13, line: 27, column: 3)
!42 = !DIDerivedType(tag: DW_TAG_typedef, name: "size_t", file: !43, line: 18, baseType: !31)
!43 = !DIFile(filename: "/opt/compiler-explorer/clang-trunk-20240726/lib/clang/20/include/__stddef_size_t.h", directory: "", checksumkind: CSK_MD5, checksum: "2c44e821a2b1951cde2eb0fb2e656867")
!44 = !DILocalVariable(name: "col", scope: !45, file: !13, line: 28, type: !42)
!45 = distinct !DILexicalBlock(scope: !46, file: !13, line: 28, column: 5)
!46 = distinct !DILexicalBlock(scope: !47, file: !13, line: 27, column: 52)
!47 = distinct !DILexicalBlock(scope: !41, file: !13, line: 27, column: 3)
!48 = !DILocation(line: 0, scope: !12)
!49 = !DILocation(line: 0, scope: !41)
!50 = !DILocation(line: 27, column: 30, scope: !47)
!51 = !{!52, !52, i64 0}
!52 = !{!"int", !53, i64 0}
!53 = !{!"omnipotent char", !54, i64 0}
!54 = !{!"Simple C/C++ TBAA"}
!55 = !DILocation(line: 27, column: 28, scope: !47)
!56 = !DILocation(line: 27, column: 3, scope: !41)
!57 = !DILocation(line: 28, column: 32, scope: !58)
!58 = distinct !DILexicalBlock(scope: !45, file: !13, line: 28, column: 5)
!59 = !DILocation(line: 28, column: 30, scope: !58)
!60 = !DILocation(line: 0, scope: !45)
!61 = !DILocation(line: 28, column: 5, scope: !45)
!62 = !{!63, !64, i64 4}
!63 = !{!"TwoDMemrefI32", !64, i64 0, !64, i64 4, !52, i64 8, !53, i64 12, !53, i64 20}
!64 = !{!"any pointer", !53, i64 0}
!65 = !DILocation(line: 35, column: 1, scope: !12)
!66 = !DILocation(line: 27, column: 48, scope: !47)
!67 = distinct !{!67, !56, !68, !69, !70}
!68 = !DILocation(line: 34, column: 3, scope: !41)
!69 = !{!"llvm.loop.mustprogress"}
!70 = !{!"llvm.loop.unswitch.partial.disable"}
!71 = !DILocation(line: 31, column: 34, scope: !72)
!72 = distinct !DILexicalBlock(scope: !58, file: !13, line: 28, column: 54)
!73 = !{!63, !52, i64 8}
!74 = !DILocation(line: 31, column: 43, scope: !72)
!75 = !DILocation(line: 31, column: 58, scope: !72)
!76 = !DILocation(line: 32, column: 35, scope: !72)
!77 = !DILocation(line: 32, column: 33, scope: !72)
!78 = !DILocation(line: 31, column: 11, scope: !72)
!79 = !DILocation(line: 29, column: 30, scope: !72)
!80 = !DILocation(line: 29, column: 39, scope: !72)
!81 = !DILocation(line: 29, column: 54, scope: !72)
!82 = !DILocation(line: 30, column: 31, scope: !72)
!83 = !DILocation(line: 30, column: 29, scope: !72)
!84 = !DILocation(line: 29, column: 7, scope: !72)
!85 = !DILocation(line: 30, column: 47, scope: !72)
!86 = !DILocation(line: 28, column: 50, scope: !58)
!87 = distinct !{!87, !61, !88, !69}
!88 = !DILocation(line: 33, column: 5, scope: !45)
!89 = distinct !DISubprogram(name: "memrefCopy8bit", scope: !13, file: !13, line: 37, type: !90, scopeLine: 37, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !105)
!90 = !DISubroutineType(types: !91)
!91 = !{null, !92, !92}
!92 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !93, size: 32)
!93 = !DIDerivedType(tag: DW_TAG_typedef, name: "TwoDMemrefI8_t", file: !13, line: 23, baseType: !94)
!94 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "TwoDMemrefI8", file: !13, line: 13, size: 224, elements: !95)
!95 = !{!96, !101, !102, !103, !104}
!96 = !DIDerivedType(tag: DW_TAG_member, name: "data", scope: !94, file: !13, line: 14, baseType: !97, size: 32)
!97 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !98, size: 32)
!98 = !DIDerivedType(tag: DW_TAG_typedef, name: "int8_t", file: !23, line: 20, baseType: !99)
!99 = !DIDerivedType(tag: DW_TAG_typedef, name: "__int8_t", file: !25, line: 41, baseType: !100)
!100 = !DIBasicType(name: "signed char", size: 8, encoding: DW_ATE_signed_char)
!101 = !DIDerivedType(tag: DW_TAG_member, name: "aligned_data", scope: !94, file: !13, line: 16, baseType: !97, size: 32, offset: 32)
!102 = !DIDerivedType(tag: DW_TAG_member, name: "offset", scope: !94, file: !13, line: 18, baseType: !29, size: 32, offset: 64)
!103 = !DIDerivedType(tag: DW_TAG_member, name: "shape", scope: !94, file: !13, line: 19, baseType: !33, size: 64, offset: 96)
!104 = !DIDerivedType(tag: DW_TAG_member, name: "stride", scope: !94, file: !13, line: 20, baseType: !33, size: 64, offset: 160)
!105 = !{!106, !107, !108, !110}
!106 = !DILocalVariable(name: "src", arg: 1, scope: !89, file: !13, line: 37, type: !92)
!107 = !DILocalVariable(name: "dst", arg: 2, scope: !89, file: !13, line: 37, type: !92)
!108 = !DILocalVariable(name: "row", scope: !109, file: !13, line: 38, type: !42)
!109 = distinct !DILexicalBlock(scope: !89, file: !13, line: 38, column: 3)
!110 = !DILocalVariable(name: "col", scope: !111, file: !13, line: 39, type: !42)
!111 = distinct !DILexicalBlock(scope: !112, file: !13, line: 39, column: 5)
!112 = distinct !DILexicalBlock(scope: !113, file: !13, line: 38, column: 52)
!113 = distinct !DILexicalBlock(scope: !109, file: !13, line: 38, column: 3)
!114 = !DILocation(line: 0, scope: !89)
!115 = !DILocation(line: 0, scope: !109)
!116 = !DILocation(line: 38, column: 30, scope: !113)
!117 = !DILocation(line: 38, column: 28, scope: !113)
!118 = !DILocation(line: 38, column: 3, scope: !109)
!119 = !DILocation(line: 39, column: 32, scope: !120)
!120 = distinct !DILexicalBlock(scope: !111, file: !13, line: 39, column: 5)
!121 = !DILocation(line: 39, column: 30, scope: !120)
!122 = !DILocation(line: 0, scope: !111)
!123 = !DILocation(line: 39, column: 5, scope: !111)
!124 = !DILocation(line: 46, column: 1, scope: !89)
!125 = !DILocation(line: 38, column: 48, scope: !113)
!126 = distinct !{!126, !118, !127, !69, !70}
!127 = !DILocation(line: 45, column: 3, scope: !109)
!128 = !DILocation(line: 42, column: 16, scope: !129)
!129 = distinct !DILexicalBlock(scope: !120, file: !13, line: 39, column: 54)
!130 = !{!131, !64, i64 4}
!131 = !{!"TwoDMemrefI8", !64, i64 0, !64, i64 4, !52, i64 8, !53, i64 12, !53, i64 20}
!132 = !DILocation(line: 42, column: 34, scope: !129)
!133 = !{!131, !52, i64 8}
!134 = !DILocation(line: 42, column: 44, scope: !129)
!135 = !DILocation(line: 42, column: 59, scope: !129)
!136 = !DILocation(line: 43, column: 36, scope: !129)
!137 = !DILocation(line: 43, column: 34, scope: !129)
!138 = !DILocation(line: 42, column: 11, scope: !129)
!139 = !{!53, !53, i64 0}
!140 = !DILocation(line: 40, column: 12, scope: !129)
!141 = !DILocation(line: 40, column: 30, scope: !129)
!142 = !DILocation(line: 40, column: 40, scope: !129)
!143 = !DILocation(line: 40, column: 55, scope: !129)
!144 = !DILocation(line: 41, column: 32, scope: !129)
!145 = !DILocation(line: 41, column: 30, scope: !129)
!146 = !DILocation(line: 40, column: 7, scope: !129)
!147 = !DILocation(line: 41, column: 49, scope: !129)
!148 = !DILocation(line: 39, column: 50, scope: !120)
!149 = distinct !{!149, !123, !150, !69}
!150 = !DILocation(line: 44, column: 5, scope: !111)

