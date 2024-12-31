	.file	1 "halt.c"
gcc2_compiled.:
__gnu_compiled_c:
	.text
	.align	2
	.globl	main
	.ent	main
main:
	.frame	$fp,40,$31		# vars= 16, regs= 2/0, args= 16, extra= 0
	.mask	0xc0000000,-4
	.fmask	0x00000000,0
	subu	$sp,$sp,40
	sw	$31,36($sp)
	sw	$fp,32($sp)
	move	$fp,$sp
	jal	__main
	li	$2,3			# 0x00000003
	sw	$2,24($fp)
	li	$2,2			# 0x00000002
	sw	$2,16($fp)
	lw	$2,16($fp)
	addu	$3,$2,-1
	sw	$3,20($fp)
	lw	$2,16($fp)
	lw	$3,20($fp)
	subu	$2,$2,$3
	lw	$3,24($fp)
	addu	$2,$3,$2
	sw	$2,24($fp)
	jal	Halt
$L1:
	move	$sp,$fp
	lw	$31,36($sp)
	lw	$fp,32($sp)
	addu	$sp,$sp,40
	j	$31
	.end	main
